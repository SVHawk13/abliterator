import functools
import re
from typing import Callable

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM

from abliterator.chat_template import LLAMA3_CHAT_TEMPLATE, ChatTemplate
from abliterator.data import prepare_dataset
from abliterator.util import batch, clear_mem, measure_fn

DEFAULT_ACTIVATION_LAYERS = ("resid_pre", "resid_post", "mlp_out", "attn_out")


class ModelAbliterator:
    def __init__(
        self,
        model: str,
        dataset: tuple[list[str], list[str]] | list[tuple[list[str], list[str]]],
        device: str = "cuda",
        n_devices: int | None = None,
        cache_fname: str | None = None,
        activation_layers: list[str] | None = None,
        chat_template: str | None = None,
        positive_toks: list[int] | tuple[int] | set[int] | Int[Tensor, "..."] = None,
        negative_toks: list[int] | tuple[int] | set[int] | Int[Tensor, "..."] = None,
        dtype: torch.dtype | str | None = None,
        hf_model: AutoModelForCausalLM | None = None,
    ) -> None:
        self.MODEL_PATH = model
        activation_layers = activation_layers or list(DEFAULT_ACTIVATION_LAYERS)
        if n_devices is None and torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
        elif n_devices is None:
            n_devices = 1

        # Save memory
        torch.set_grad_enabled(False)

        model_options = {
            "model_name": model,
            "n_devices": n_devices,
            "device": device,
            "dtype": dtype or "float32",
            "default_padding_side": "left",
            "hf_model": hf_model,
        }

        self.model = HookedTransformer.from_pretrained_no_processing(**model_options)

        self.model.requires_grad_(False)

        self.model.tokenizer.padding_side = "left"
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.chat_template = chat_template or ChatTemplate(self, LLAMA3_CHAT_TEMPLATE)

        self.hidden_size = self.model.cfg.d_model
        self.original_state = {
            k: v.to("cpu") for k, v in self.model.state_dict().items()
        }
        self.harmful = {}
        self.harmless = {}
        self.modified_layers = {"mlp": {}, "W_O": {}}
        self.checkpoints = []

        if cache_fname is not None:
            outs = torch.load(cache_fname, map_location="cpu")
            self.harmful, self.harmless, modified_layers, checkpoints = outs[:4]
            self.checkpoints = checkpoints or []
            self.modified_layers = modified_layers

        self.harmful_inst_train, self.harmful_inst_test = prepare_dataset(dataset[0])
        self.harmless_inst_train, self.harmless_inst_test = prepare_dataset(dataset[1])

        self.fwd_hooks = []
        self.modified = False
        self.activation_layers = (
            [activation_layers]
            if isinstance(activation_layers, str)
            else activation_layers
        )
        if negative_toks is None:
            print(
                "WARNING: You've not set 'negative_toks', defaulting to tokens for Llama-3 vocab"
            )
            self.negative_toks = {
                4250,
                14931,
                89735,
                20451,
                11660,
                11458,
                956,
            }  # llama-3 refusal tokens e.g. ' cannot', ' unethical', ' sorry'
        else:
            self.negative_toks = negative_toks
        if positive_toks is None:
            print(
                "WARNING: You've not set 'positive_toks', defaulting to tokens for Llama-3 vocab"
            )
            self.positive_toks = {32, 1271, 8586, 96556, 78145}
        else:
            self.positive_toks = positive_toks
        self._blacklisted = set()

    def __enter__(self):
        if hasattr(self, "current_state"):
            raise Exception("Cannot do multi-contexting")
        self.current_state = self.model.state_dict()
        self.current_layers = self.modified_layers.copy()
        self.was_modified = self.modified
        return self

    def __exit__(self, exc, exc_value, exc_tb):
        self.model.load_state_dict(self.current_state)
        del self.current_state
        self.modified_layers = self.current_layers
        del self.current_layers
        self.modified = self.was_modified
        del self.was_modified

    def reset_state(self):
        self.modified = False
        self.modified_layers = {"mlp": {}, "W_O": {}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self):
        # MAYBE: Offload to disk? That way we're not taking up RAM with this
        self.checkpoints.append(self.modified_layers.copy())

    # Utility functions

    def blacklist_layer(self, layer: int | list[int]):
        # Prevents a layer from being modified
        if isinstance(layer, list):
            self._blacklisted.update(layer)
        else:
            self._blacklisted.add(layer)

    def whitelist_layer(self, layer: int | list[int]):
        # Removes layer from blacklist to allow modification
        if isinstance(layer, list):
            self._blacklisted.difference_update(layer)
        else:
            self._blacklisted.discard(layer)

    def save_activations(self, fname: str):
        torch.save(
            [
                self.harmful,
                self.harmless,
                self.modified_layers
                if self.modified_layers["mlp"] or self.modified_layers["W_O"]
                else None,
                self.checkpoints if len(self.checkpoints) > 0 else None,
            ],
            fname,
        )

    def get_whitelisted_layers(self) -> list[int]:
        return [l for l in range(self.model.cfg.n_layers) if l not in self._blacklisted]

    def get_all_act_names(
        self, activation_layers: list[str] | None = None
    ) -> list[tuple[int, str]]:
        return [
            (i, utils.get_act_name(act_name, i))
            for i in self.get_whitelisted_layers()
            for act_name in (activation_layers or self.activation_layers)
        ]

    def calculate_mean_dirs(
        self, key: str, include_overall_mean: bool = False
    ) -> dict[str, Float[Tensor, "d_model"]]:
        dirs = {
            "harmful_mean": torch.mean(self.harmful[key], dim=0),
            "harmless_mean": torch.mean(self.harmless[key], dim=0),
        }

        if include_overall_mean:
            if (
                self.harmful[key].shape != self.harmless[key].shape
                or self.harmful[key].device.type == "cuda"
            ):
                # If the shapes are different, we can't add them together; we'll need to concatenate the tensors first.
                # Using 'cpu', this is slower than the alternative below.
                # Using 'cuda', this seems to be faster than the alternatives.
                # NOTE: Assume both tensors are on the same device.
                #
                dirs["mean_dir"] = torch.mean(
                    torch.cat((self.harmful[key], self.harmless[key]), dim=0), dim=0
                )
            else:
                # If the shapes are the same, we can add them together, take the mean,
                # then divide by 2.0 to account for the initial element-wise addition of the tensors.
                #
                # The result is identical to:
                #    `torch.sum(self.harmful[key] + self.harmless[key]) / (len(self.harmful[key]) + len(self.harmless[key]))`
                #
                dirs["mean_dir"] = (
                    torch.mean(self.harmful[key] + self.harmless[key], dim=0) / 2.0
                )

        return dirs

    def calculate_scaled_projection(
        self,
        components: Float[Tensor, "... d_model"],
        direction: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "... d_model"]:
        return (
            einops.einsum(
                components,
                direction.view(-1, 1),
                "... d_model, d_model column -> ... column",
            )
            * direction
        )

    def calculate_ortho_complement(
        self,
        components: Float[Tensor, "... d_model"],
        direction: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "... d_model"]:
        return components - self.calculate_scaled_projection(components, direction)

    def get_avg_projections(
        self, key: str, direction: Float[Tensor, "d_model"]
    ) -> tuple[Float[Tensor, "d_model"], Float[Tensor, "d_model"]]:
        dirs = self.calculate_mean_dirs(self, key)
        return (
            torch.dot(dirs["harmful_mean"], direction),
            torch.dot(dirs["harmless_mean"], direction),
        )

    def get_layer_dirs(
        self, layer, key: str | None = None, include_overall_mean: bool = False
    ) -> dict[str, Float[Tensor, "d_model"]]:
        act_key = key or self.activation_layers[0]
        if len(self.harmfuls[key]) < layer:
            raise IndexError("Invalid layer")
        return self.calculate_mean_dirs(
            utils.get_act_name(act_key, layer),
            include_overall_mean=include_overall_mean,
        )

    def ortho_complement_hook(
        self,
        activation: Float[Tensor, "... d_model"],
        hook: HookPoint,
        direction: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "... d_model"]:
        if activation.device != direction.device:
            direction = direction.to(activation.device)
        return self.calculate_ortho_complement(activation, direction)

    def refusal_dirs(self, invert: bool = False) -> dict[str, Float[Tensor, "d_model"]]:
        if not self.harmful:
            raise IndexError("No cache")

        refusal_dirs = {
            key: self.calculate_mean_dirs(key)
            for key in self.harmful
            if ".0." not in key
        }  # don't include layer 0, as it often becomes NaN
        if invert:
            refusal_dirs = {
                key: v["harmless_mean"] - v["harmful_mean"]
                for key, v in refusal_dirs.items()
            }
        else:
            refusal_dirs = {
                key: v["harmful_mean"] - v["harmless_mean"]
                for key, v in refusal_dirs.items()
            }

        return {key: (v / v.norm()).to("cpu") for key, v in refusal_dirs.items()}

    def scored_dirs(self, invert=False) -> list[tuple[str, Float[Tensor, "d_model"]]]:
        refusals = self.refusal_dirs(invert=invert)
        return sorted(
            [(ln, refusals[act_name]) for ln, act_name in self.get_all_act_names()],
            reverse=True,
            key=lambda x: abs(x[1].mean()),
        )

    def get_layer_of_act_name(self, ref: str) -> str | int:
        s = re.search(r"\.(\d+)\.", ref)
        return s if s is None else int(s[1])

    def layer_attn(
        self, layer: int, replacement: Float[Tensor, "d_model"] = None
    ) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            # make sure device doesn't change
            self.modified = True
            self.model.blocks[layer].attn.W_O.data = replacement.to(
                self.model.blocks[layer].attn.W_O.device
            )
            self.modified_layers["W_O"][layer] = [
                *self.modified_layers.get(layer, []),
                (
                    self.model.blocks[layer].attn.W_O.data.to("cpu"),
                    replacement.to("cpu"),
                ),
            ]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(
        self, layer: int, replacement: Float[Tensor, "d_model"] = None
    ) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            # make sure device doesn't change
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(
                self.model.blocks[layer].mlp.W_out.device
            )
            self.modified_layers["mlp"][layer] = [
                *self.modified_layers.get(layer, []),
                (
                    self.model.blocks[layer].mlp.W_out.data.to("cpu"),
                    replacement.to("cpu"),
                ),
            ]
        return self.model.blocks[layer].mlp.W_out.data

    def tokenize_instructions_fn(
        self, instructions: list[str]
    ) -> Int[Tensor, "batch_size seq_len"]:
        prompts = [
            self.chat_template.format(instruction=instruction)
            for instruction in instructions
        ]
        return self.model.tokenizer(
            prompts, padding=True, truncation=False, return_tensors="pt"
        ).input_ids

    def generate_logits(
        self,
        toks: Int[Tensor, "batch_size seq_len"],
        *args,
        drop_refusals: bool = True,
        stop_at_eos: bool = False,
        max_tokens_generated: int = 1,
        **kwargs,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len d_vocab"], Int[Tensor, "batch_size seq_len"]
    ]:
        # does most of the model magic
        all_toks = torch.zeros(
            (toks.shape[0], toks.shape[1] + max_tokens_generated),
            dtype=torch.long,
            device=toks.device,
        )
        all_toks[:, : toks.shape[1]] = toks
        generating = list(range(toks.shape[0]))
        for i in range(max_tokens_generated):
            logits = self.model(
                all_toks[generating, : -max_tokens_generated + i], *args, **kwargs
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1).to("cpu")
            all_toks[generating, -max_tokens_generated + i] = next_tokens
            if drop_refusals and any(
                negative_tok in next_tokens for negative_tok in self.negative_toks
            ):
                # refusals we handle differently: if it's misbehaving, we stop all batches and move on to the next one
                break
            if stop_at_eos:
                for batch_idx in generating:
                    generating = [
                        i
                        for i in range(toks.shape[0])
                        if all_toks[i][-1] != self.model.tokenizer.eos_token_id
                    ]
                if len(generating) == 0:
                    break
        return logits, all_toks

    def generate(
        self,
        prompt: list[str] | str,
        *model_args,
        max_tokens_generated: int = 64,
        stop_at_eos: bool = True,
        **model_kwargs,
    ) -> list[str]:
        # convenience function to test manual prompts, no caching
        if type(prompt) is str:
            gen = self.tokenize_instructions_fn([prompt])
        else:
            gen = self.tokenize_instructions_fn(prompt)

        logits, all_toks = self.generate_logits(
            gen,
            *model_args,
            stop_at_eos=stop_at_eos,
            max_tokens_generated=max_tokens_generated,
            **model_kwargs,
        )
        return self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)

    def test(
        self,
        *args,
        test_set: list[str] | None = None,
        N: int = 16,
        batch_size: int = 4,
        **kwargs,
    ):
        if test_set is None:
            test_set = self.harmful_inst_test
        for prompts in batch(test_set[: min(len(test_set), N)], batch_size):
            for res in self.generate(prompts, *args, **kwargs):
                print(res)

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] | None = None,
        incl_bwd: bool = False,
        device: str | None = None,
        remove_batch_dim: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        fwd_hooks: list[str] | None = None,
        max_new_tokens: int = 1,
        **model_kwargs,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len d_vocab"],
        dict[str, Float[Tensor, "batch_size seq_len d_model"]],
    ]:
        fwd_hooks = fwd_hooks or []
        if names_filter is None and self.activation_layers:

            def activation_layering(namefunc: str):
                return any(s in namefunc for s in self.activation_layers)

            names_filter = activation_layering

        cache_dict, fwd, bwd = self.model.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=utils.Slice(None),
        )

        fwd_hooks = fwd_hooks + fwd + self.fwd_hooks

        if not max_new_tokens:
            # must do at least 1 token
            max_new_tokens = 1

        with self.model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            # model_out = self.model(*model_args,**model_kwargs)
            model_out, toks = self.generate_logits(
                *model_args, max_tokens_generated=max_new_tokens, **model_kwargs
            )
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def apply_refusal_dirs(
        self,
        refusal_dirs: list[Float[Tensor, "d_model"]],
        W_O: bool = True,
        mlp: bool = True,
        layers: list[str] | None = None,
    ):
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for refusal_dir in refusal_dirs:
            for layer in layers:
                for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                    if not modifying[0]:
                        continue
                    matrix = modifying[1](layer)
                    if refusal_dir.device != matrix.device:
                        refusal_dir = refusal_dir.to(matrix.device)
                    proj = self.calculate_scaled_projection(matrix, refusal_dir)
                    modifying[1](layer, matrix - proj)

    def induce_refusal_dir(
        self,
        refusal_dir: Float[Tensor, "d_model"],
        W_O: bool = True,
        mlp: bool = True,
        layers: list[str] | None = None,
    ):
        # incomplete, needs work
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for layer in layers:
            for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                if modifying[0]:
                    matrix = modifying[1](layer)
                    if refusal_dir.device != matrix.device:
                        refusal_dir = refusal_dir.to(matrix.device)
                    proj = self.calculate_scaled_projection(matrix, refusal_dir)
                    avg_proj = refusal_dir * self.get_avg_projections(
                        utils.get_act_name(self.activation_layers[0], layer),
                        refusal_dir,
                    )
                    modifying[1](layer, (matrix - proj) + avg_proj)

    def test_dir(
        self,
        refusal_dir: Float[Tensor, "d_model"],
        activation_layers: list[str] | None = None,
        use_hooks: bool = True,
        layers: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Float[Tensor, "d_model"]]:
        # `use_hooks=True` is better for bigger models as it causes a lot of memory swapping otherwise, but
        # `use_hooks=False` is much more representative of the final weights manipulation

        before_hooks = self.fwd_hooks
        try:
            if layers is None:
                layers = self.get_whitelisted_layers()

            if activation_layers is None:
                activation_layers = self.activation_layers

            if use_hooks:
                hooks = self.fwd_hooks
                hook_fn = functools.partial(
                    self.ortho_complement_hook, direction=refusal_dir
                )
                self.fwd_hooks = before_hooks + [
                    (act_name, hook_fn) for ln, act_name in self.get_all_act_names()
                ]
                return self.measure_scores(**kwargs)
            else:
                with self:
                    self.apply_refusal_dirs([refusal_dir], layers=layers)
                    return self.measure_scores(**kwargs)
        finally:
            self.fwd_hooks = before_hooks

    def find_best_refusal_dir(
        self,
        N: int = 4,
        positive: bool = False,
        use_hooks: bool = True,
        invert: bool = False,
    ) -> list[tuple[float, str]]:
        dirs = self.refusal_dirs(invert=invert)
        if self.modified:
            print(
                "WARNING: Modified; will restore model to current modified state each run"
            )
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1], N=N, use_hooks=use_hooks)[int(positive)]
            scores.append((score, direction))
        return sorted(scores, key=lambda x: x[0])

    def measure_scores(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = "max",
        batch_measure: str = "max",
        positive: bool = False,
    ) -> dict[str, Float[Tensor, "d_model"]]:
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_test[:N])
        logits, cache = self.run_with_cache(
            toks, max_new_tokens=sampled_token_ct, drop_refusals=False
        )

        negative_score, positive_score = self.measure_scores_from_logits(
            logits, sampled_token_ct, measure=batch_measure
        )

        negative_score = measure_fn(measure, negative_score)
        positive_score = measure_fn(measure, positive_score)
        return {
            "negative": negative_score.to("cpu"),
            "positive": positive_score.to("cpu"),
        }

    def measure_scores_from_logits(
        self,
        logits: Float[Tensor, "batch_size seq_len d_vocab"],
        sequence: int,
        measure: str = "max",
    ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        normalized_scores = torch.softmax(logits[:, -sequence:, :].to("cpu"), dim=-1)[
            :, :, list(self.positive_toks) + list(self.negative_toks)
        ]

        normalized_positive, normalized_negative = torch.split(
            normalized_scores, [len(self.positive_toks), len(self.negative_toks)], dim=2
        )

        max_negative_score_per_sequence = torch.max(normalized_negative, dim=-1)[0]
        max_positive_score_per_sequence = torch.max(normalized_positive, dim=-1)[0]

        negative_score_per_batch = measure_fn(
            measure, max_negative_score_per_sequence, dim=-1
        )[0]
        positive_score_per_batch = measure_fn(
            measure, max_positive_score_per_sequence, dim=-1
        )[0]
        return negative_score_per_batch, positive_score_per_batch

    def do_resid(
        self, fn_name: str
    ) -> tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        list[str],
    ]:
        if not any("resid" in k for k in self.harmless.keys()):
            raise AssertionError(
                "You need residual streams to decompose layers! Run cache_activations with None in `activation_layers`"
            )
        resid_harmful, labels = getattr(self.harmful, fn_name)(
            apply_ln=True, return_labels=True
        )
        resid_harmless = getattr(self.harmless, fn_name)(apply_ln=True)

        return resid_harmful, resid_harmless, labels

    def decomposed_resid(
        self,
    ) -> tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        list[str],
    ]:
        return self.do_resid("decompose_resid")

    def accumulated_resid(
        self,
    ) -> tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        list[str],
    ]:
        return self.do_resid("accumulated_resid")

    def unembed_resid(
        self, resid: Float[Tensor, "layer batch d_model"], pos: int = -1
    ) -> Float[Tensor, "layer batch d_vocab"]:
        W_U = self.model.W_U
        if pos is None:
            return einops.einsum(
                resid.to(W_U.device),
                W_U,
                "layer batch d_model, d_model d_vocab -> layer batch d_vocab",
            ).to("cpu")
        else:
            return einops.einsum(
                resid[:, pos, :].to(W_U.device),
                W_U,
                "layer d_model, d_model d_vocab -> layer d_vocab",
            ).to("cpu")

    def create_layer_rankings(
        self,
        token_set: list[int] | set[int] | Int[Tensor, "..."],
        decompose: bool = True,
        token_set_b: list[int] | set[int] | Int[Tensor, "..."] = None,
    ) -> list[tuple[int, int]]:
        decomposer = self.decomposed_resid if decompose else self.accumulated_resid

        decomposed_resid_harmful, decomposed_resid_harmless, labels = decomposer()

        W_U = self.model.W_U.to("cpu")
        unembedded_harmful = self.unembed_resid(decomposed_resid_harmful)
        unembedded_harmless = self.unembed_resid(decomposed_resid_harmless)

        sorted_harmful_indices = torch.argsort(
            unembedded_harmful, dim=1, descending=True
        )
        sorted_harmless_indices = torch.argsort(
            unembedded_harmless, dim=1, descending=True
        )

        harmful_set = torch.isin(sorted_harmful_indices, torch.tensor(list(token_set)))
        harmless_set = torch.isin(
            sorted_harmless_indices,
            torch.tensor(list(token_set if token_set_b is None else token_set_b)),
        )

        indices_in_set = zip(
            harmful_set.nonzero(as_tuple=True)[1],
            harmless_set.nonzero(as_tuple=True)[1],
        )
        return indices_in_set

    def mse_positive(
        self, N: int = 128, batch_size: int = 8, last_indices: int = 1
    ) -> dict[str, Float[Tensor, "d_model"]]:
        # Calculate mean squared error against currently loaded negative cached activation
        # Idea being to get a general sense of how the "normal" direction has been altered.
        # This is to compare ORIGINAL functionality to ABLATED functionality, not for ground truth.

        # load full training set to ensure alignment
        toks = self.tokenize_instructions_fn(
            instructions=self.harmful_inst_train[:N] + self.harmless_inst_train[:N]
        )

        splitpos = min(N, len(self.harmful_inst_train))

        # select for just harmless
        toks = toks[splitpos:]
        self.loss_harmless = {}

        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(
                toks[i : min(i + batch_size, len(toks))]
            )
            for key in cache:
                if any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :], dim=1).to(
                        "cpu"
                    )
                    if key not in self.loss_harmless:
                        self.loss_harmless[key] = tensor
                    else:
                        self.loss_harmless[key] = torch.cat(
                            (self.loss_harmless[key], tensor), dim=0
                        )
            del logits, cache
            clear_mem()

        return {
            k: F.mse_loss(
                self.loss_harmless[k].float()[:N], self.harmless[k].float()[:N]
            )
            for k in self.loss_harmless
        }

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        measure_refusal: int = 0,
        stop_at_layer: int | None = None,
    ) -> tuple[ActivationCache, list[str]]:
        # Base functionality for creating an activation cache with a training set, prefer 'cache_activations' for regular usage

        base = {}
        z_label = [] if measure_refusal > 1 else None
        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(
                toks[i : min(i + batch_size, len(toks))],
                max_new_tokens=measure_refusal,
                stop_at_layer=stop_at_layer,
            )

            if measure_refusal > 1:
                z_label.extend(
                    self.measure_scores_from_logits(logits, measure_refusal)[0]
                )
            for key in cache:
                if self.activation_layers is None or any(
                    k in key for k in self.activation_layers
                ):
                    tensor = torch.mean(
                        cache[key][:, -last_indices:, :].to("cpu"), dim=1
                    )
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)

            del logits, cache
            clear_mem()

        return ActivationCache(base, self.model), z_label

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        measure_refusal: int = 0,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_harmless: bool = True,
        stop_at_layer: int | None = None,
    ):
        if hasattr(self, "current_state"):
            print("WARNING: Caching activations using a context")
        if self.modified:
            print("WARNING: Running modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        harmless_is_set = len(getattr(self, "harmless", {})) > 0
        preserve_harmless = harmless_is_set and preserve_harmless

        if reset is True or getattr(self, "harmless", None) is None:
            self.harmful = {}
            if not preserve_harmless:
                self.harmless = {}

            self.harmful_z_label = []
            self.harmless_z_label = []

        # load the full training set here to align all the dimensions (even if we're not going to run harmless)
        toks = self.tokenize_instructions_fn(
            instructions=self.harmful_inst_train[:N] + self.harmless_inst_train[:N]
        )

        splitpos = min(N, len(self.harmful_inst_train))
        harmful_toks = toks[:splitpos]
        harmless_toks = toks[splitpos:]

        last_indices = last_indices or 1

        self.harmful, self.harmful_z_label = self.create_activation_cache(
            harmful_toks,
            N=N,
            batch_size=batch_size,
            last_indices=last_indices,
            measure_refusal=measure_refusal,
            stop_at_layer=None,
        )
        if not preserve_harmless:
            self.harmless, self.harmless_z_label = self.create_activation_cache(
                harmless_toks,
                N=N,
                batch_size=batch_size,
                last_indices=last_indices,
                measure_refusal=measure_refusal,
                stop_at_layer=None,
            )

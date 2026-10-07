import inspect
import sys
from typing import Optional

import torch
from torch import Tensor, nn

from prometheo.models.pooling import PoolingMethods
from worldcereal.openeo.inference import SeasonalInferenceEngine
from cropland_croptype_mapping_local import parse_args
from cropland_croptype_mapping_local import DEFAULT_SEASON_WINDOWS
from cropland_croptype_mapping_local import build_postprocess_spec
from cropland_croptype_mapping_local import parse_manual_windows
from cropland_croptype_mapping_local import load_input_cube
import xarray as xr

from worldcereal.train.seasonal_head import WorldCerealSeasonalModel


class WorldCerealSeasonalONNX(nn.Module):
    """ONNX-friendly wrapper: pure-tensor forward, tuple output."""

    def __init__(self, model: WorldCerealSeasonalModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month= 0,
        season_masks = None
    ):

        time_embeddings = self.model.backbone.encoder(
            x,dynamic_world,latlons,mask,month, eval_pooling=PoolingMethods.TIME.value
        )
        #Rearrange from:  prometheo/prometheo/models/presto/wrapper.py:548
        # from einops import rearrange
        # w = 16
        # h = 32
        # b = time_embeddings.shape[0] / (h * w)
        # embeddings = rearrange(
        #     time_embeddings, "(b h w) t d -> b h w t d", b=b.int(), h=h, w=w
        # )

        # flat = WorldCerealSeasonalModel._flatten_embeddings(time_embeddings)
        flat = time_embeddings
        out = self.model.head(flat, season_masks)

        # Return only tensors. Use zeros as placeholders if a head is disabled,
        # OR build two separate wrappers (croptype-only / landcover-only).
        global_logits = out.global_logits if out.global_logits is not None \
            else torch.zeros(flat.shape[0], 0, device=flat.device)
        season_logits = out.season_logits if out.season_logits is not None \
            else torch.zeros(flat.shape[0], season_masks.shape[1], 0, device=flat.device)

        return out.global_embedding, out.season_embeddings, torch.softmax(global_logits,-1), torch.softmax(season_logits,-1)

def export_onnx_model(engine, season_windows,season_ids):
    import pickle


    input_names = ["x", "dynamic_world", "latlons", "mask", "month", "season_masks"]

    from torch.export.dynamic_shapes import Dim
    dynamic = {"x": (Dim.DYNAMIC, Dim.STATIC, Dim.STATIC), "dynamic_world": (Dim.DYNAMIC, Dim.STATIC),
               "latlons": (Dim.DYNAMIC, Dim.STATIC), "mask": (Dim.DYNAMIC, Dim.STATIC, Dim.STATIC),
               "month": (Dim.DYNAMIC, Dim.STATIC), "eval_pooling": None}
    dynamic_axes = {"x": {0: "batch_size"}, "dynamic_world": {0: "batch_size"},
                    "latlons": {0: "batch_size"}, "mask": {0: "batch_size"},
                    "month": {0: "batch_size"},"season_masks": {0: "batch_size"}}

    #this is a pickled version of the python object going in to the prometheo model, after conversion from the predictors
    presto_args = pickle.load(open("./presto_args_worldcereal.pkl", "rb"))
    presto_args[1]['eval_pooling'] = 'time'

    arr = xr.open_dataarray("../../tests/worldcerealtests/testresources/test_inference_array.nc")
    import torch
    timestamps = arr.t.values

    disable_latlons = True
    if disable_latlons:
        presto_args[1]['latlons'] = None
        input_names.remove("latlons")
        del dynamic_axes['latlons']
        del dynamic['latlons']


    season_masks = engine._resolve_season_masks(
                timestamps=timestamps,
                batch_size=512,
                season_windows=season_windows,
                season_masks=None,
                season_ids=season_ids,
            )
    presto_args[1]['season_masks'] = torch.from_numpy(season_masks[0])
    # if using dyname=True => dynamic_shapes = dynamic
    wrapper = WorldCerealSeasonalONNX(engine.bundle.model).eval()

    onnx_program = torch.onnx.export(wrapper, presto_args, f="worldcereal_seasonal_us_nolatlon.onnx",
                                     input_names=input_names, dynamic_axes=dynamic_axes, dynamo=False, verbose=True)


def main() -> None:
    default_arg_list = [
        "--season-window",
        DEFAULT_SEASON_WINDOWS[0],
        "--season-window",
        DEFAULT_SEASON_WINDOWS[1],
        "--memory-logging",
        "--cropland-postprocess",
        "--croptype-postprocess",
        # "--export-class-probabilities",
    ]
    arg_list = None if len(sys.argv) > 1 else default_arg_list
    args = parse_args(arg_list)

    try:
        arr, template = load_input_cube(args.input, args.sample_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load input cube: {exc}", file=sys.stderr)
        sys.exit(1)

    season_ids = args.season_id if args.season_id else None

    # If the user passed custom CLI args but omitted season windows,
    # fall back to a known-valid two-season setup for local testing.
    if args.season_window is None and season_ids is None:
        args.season_window = list(DEFAULT_SEASON_WINDOWS)
        print(
            "No --season-window provided; using default windows for tc-s1/tc-s2.",
            file=sys.stderr,
        )
    try:
        season_windows = parse_manual_windows(args.season_window)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    cropland_postprocess = build_postprocess_spec(
        enabled=args.cropland_postprocess,
        method=args.cropland_method,
        kernel_size=args.cropland_kernel_size,
    )
    croptype_postprocess = build_postprocess_spec(
        enabled=args.croptype_postprocess,
        method=args.croptype_method,
        kernel_size=args.croptype_kernel_size,
    )

    engine_kwargs = {
        "seasonal_model_zip": args.seasonal_zip,
        "landcover_head_zip": args.landcover_head_zip,
        "croptype_head_zip": args.croptype_head_zip,
        "cache_root": args.cache_dir,
        "batch_size": args.batch_size,
        "season_ids": season_ids,
        "export_class_probabilities": args.export_class_probabilities,
        "cropland_postprocess": cropland_postprocess,
        "croptype_postprocess": croptype_postprocess,
    }
    init_params = inspect.signature(SeasonalInferenceEngine.__init__).parameters
    if "auto_tune_batch_size" in init_params:
        engine_kwargs["auto_tune_batch_size"] = args.auto_tune_batch_size
    elif args.auto_tune_batch_size:
        print(
            "Warning: installed SeasonalInferenceEngine does not support auto_tune_batch_size.",
            file=sys.stderr,
        )

    if "cpu_num_threads" in init_params and args.cpu_num_threads is not None:
        engine_kwargs["cpu_num_threads"] = args.cpu_num_threads
    elif args.cpu_num_threads is not None:
        print(
            "Warning: installed SeasonalInferenceEngine does not support cpu_num_threads.",
            file=sys.stderr,
        )

    if (
        "cpu_num_interop_threads" in init_params
        and args.cpu_num_interop_threads is not None
    ):
        engine_kwargs["cpu_num_interop_threads"] = args.cpu_num_interop_threads
    elif args.cpu_num_interop_threads is not None:
        print(
            "Warning: installed SeasonalInferenceEngine does not support cpu_num_interop_threads.",
            file=sys.stderr,
        )

    if "memory_logging" in init_params:
        engine_kwargs["memory_logging"] = args.memory_logging
        if "memory_logging_verbose" in init_params:
            engine_kwargs["memory_logging_verbose"] = args.memory_logging_verbose
        if "memory_report_top_n" in init_params:
            engine_kwargs["memory_report_top_n"] = args.memory_report_top_n
    elif args.memory_logging:
        print(
            "Warning: installed SeasonalInferenceEngine does not support memory_logging."
            " Use the local source tree or reinstall worldcereal from this checkout.",
            file=sys.stderr,
        )

    engine = SeasonalInferenceEngine(**engine_kwargs)

    export_onnx_model(engine, season_windows or None, season_ids)



if __name__ == "__main__":
    main()
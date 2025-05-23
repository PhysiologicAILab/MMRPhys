import torch
import onnx
import json
import logging
from pathlib import Path
from tools.torch2onnx.weight_mapper import convert_weights


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnnxConverter:
    def __init__(self, model_path, onnx_path, config_path, num_frames=181,
                 num_channels=3, height=9, width=9, half_precision=False):
        self.model_path = Path(model_path)
        self.onnx_path = Path(onnx_path)
        self.config_path = Path(config_path)
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.half_precision = half_precision

        if self.height == 9:
            from tools.torch2onnx.MMRPhysSEF import MMRPhysSEF as rPhysModel
        elif self.height == 72:
            from tools.torch2onnx.MMRPhysLEF import MMRPhysLEF as rPhysModel
        else:
            raise ValueError("Invalid height. Supported values: 9, 72")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load or create config
        self.config = self._load_or_create_config()

        # Initialize model
        try:
            # Create model
            self.model = rPhysModel(
                frames=self.num_frames,
                md_config=self.config,
                in_channels=self.num_channels,
                device=self.device
            )

            # Load weights
            convert_weights(self.model_path, self.model)

            # Prepare for conversion
            self.model.eval()
            self.model.to(self.device)
            
            # Convert model to half precision if requested
            if self.half_precision:
                logger.info("Converting model to half precision (FP16)")
                self.model.half()

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise


    def _load_or_create_config(self):
        """Load or create model configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info("Loaded existing config")
                return config

        config = {
            "FRAME_NUM": self.num_frames,
            "TASKS": ["BVP", "RSP"],
            "FS": 30,
            "sampling_rate": 30,
            "input_size": [1, self.num_channels, self.num_frames, self.height, self.width],
            "output_names": ["rPPG", "rRSP"],
            "model_info": {
                "name": "SCAMPS_Multi",
                "version": "1.0",
                "description": "Remote physiological signal extraction model"
            },
            "signal_parameters": {
                "bvp": {
                    "min_rate": 40,
                    "max_rate": 180,
                    "buffer_size": 180
                },
                "resp": {
                    "min_rate": 8,
                    "max_rate": 30,
                    "buffer_size": 180
                }
            },
            "half_precision": self.half_precision
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Created new config file")
        return config


    def _force_eval_mode(self):
        """Ensure model is in evaluation mode"""
        def _set_eval(module):
            if hasattr(module, 'train'):
                module.train(False)
            if isinstance(module, torch.nn.InstanceNorm3d):
                module.track_running_stats = False
            for param in module.parameters():
                param.requires_grad = False

        self.model.apply(_set_eval)

    def convert(self):
        """Convert model to ONNX format"""
        try:
            # Ensure eval mode
            self._force_eval_mode()

            # Create dummy input
            dummy_input = torch.randn(
                1, self.num_channels, self.num_frames,
                self.height, self.width,
                requires_grad=False
            ).to(self.device)
            
            # Convert input to half precision if model is in half precision
            if self.half_precision:
                dummy_input = dummy_input.half()

            # Define dynamic axes
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'frames'},
                'rPPG': {0: 'batch_size', 2: 'frames'},
                'rRSP': {0: 'batch_size', 2: 'frames'}
            }

            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['rPPG', 'rRSP'],
                dynamic_axes=dynamic_axes,
                verbose=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False
            )

            # Add half precision info to ONNX model metadata
            if self.half_precision:
                model = onnx.load(self.onnx_path)
                if not model.metadata_props:
                    model.metadata_props.add(key="precision", value="float16")
                else:
                    prop = model.metadata_props.add()
                    prop.key = "precision"
                    prop.value = "float16"
                onnx.save(model, self.onnx_path)

            logger.info(f"Model exported to {self.onnx_path}")
            self._verify_onnx()

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

    def _verify_onnx(self):
        """Verify the exported ONNX model"""
        try:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model, full_check=True)
            logger.info("ONNX model verification passed")

            # Print model graph
            logger.info("\nONNX Model Graph:")
            for node in model.graph.node:
                logger.info(f"Op Type: {node.op_type}")
                logger.info(f"Inputs: {node.input}")
                logger.info(f"Outputs: {node.output}")
                logger.info("---")

        except Exception as e:
            logger.error(f"ONNX verification failed: {str(e)}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=181)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--half_precision', action='store_true', 
                        help='Convert to half precision (FP16) model')

    args = parser.parse_args()

    converter = OnnxConverter(
        args.model_path,
        args.onnx_path,
        args.config_path,
        args.num_frames,
        args.num_channels,
        args.height,
        args.width,
        args.half_precision
    )
    converter.convert()
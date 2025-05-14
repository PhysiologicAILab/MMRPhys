import argparse
import json
import logging
from pathlib import Path
from tools.torch2onnx.convert_to_onnx import OnnxConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and validate the configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['input_size', 'FRAME_NUM']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")
            
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in config file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to ONNX using configuration from JSON file')
    parser.add_argument('config_path', type=str,
                        help='Path to the JSON configuration file')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--half_precision', action='store_true', default=False,
                        help='Convert to half precision (FP16) model')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load and validate config
        logger.info(f"Loading configuration from {args.config_path}")
        config = load_config(args.config_path)

        # Extract parameters from config
        input_size = config['input_size']
        if len(input_size) != 5:
            raise ValueError("input_size must have 5 dimensions [batch, channels, frames, height, width]")

        num_frames = config['FRAME_NUM']
        num_channels = input_size[1]
        height = input_size[3]
        width = input_size[4]
        
        # Check for half precision in config (command line argument takes precedence)
        half_precision = args.half_precision or config.get('half_precision', False)
        if half_precision:
            logger.info("Half precision (FP16) conversion enabled")

        # Construct paths
        config_dir = Path(args.config_path).parent
        model_name = config.get('model_info', {}).get('name', 'SCAMPS_Multi')
        
        model_path = config_dir / f"{model_name}.pth"
        onnx_path = config_dir / f"{model_name}{'_fp16' if half_precision else ''}.onnx"

        # Create output directories if they don't exist
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize converter
        logger.info("Initializing converter...")
        converter = OnnxConverter(
            model_path=str(model_path),
            onnx_path=str(onnx_path),
            config_path=args.config_path,
            num_frames=num_frames,
            num_channels=num_channels,
            height=height,
            width=width,
            half_precision=half_precision
        )

        # Perform conversion
        logger.info("Starting conversion process...")
        converter.convert()

        logger.info("Conversion completed successfully!")
        logger.info(f"ONNX model saved to: {onnx_path}")
        logger.info(f"Using config file: {args.config_path}")
        if half_precision:
            logger.info("Model converted to half precision (FP16)")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
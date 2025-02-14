import argparse
import logging
from pathlib import Path
from torch2onnx.convert_to_onnx import OnnxConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to ONNX with weight mapping')
    parser.add_argument('--model_path', type=str,
                        default='torch2onnx/SCAMPS_MMRPhysSEF_BVP_RSP_RGBx180x9_SFSAM_Label_Epoch0.pth')
    parser.add_argument('--onnx_path', type=str,
                        default='torch2onnx/SCAMPS_Multi_9x9.onnx')
    parser.add_argument('--config_path', type=str,
                        default='torch2onnx/config.json')
    parser.add_argument('--temp_weights', type=str,
                        default='torch2onnx/temp_converted_weights.pth')
    parser.add_argument('--num_frames', type=int, default=181)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create output directories if they don't exist
        onnx_path = Path(args.onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = Path(args.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize converter
        logger.info("Initializing converter...")
        converter = OnnxConverter(
            model_path=args.model_path,
            onnx_path=args.onnx_path,
            config_path=args.config_path,
            num_frames=args.num_frames,
            num_channels=args.num_channels,
            height=args.height,
            width=args.width
        )

        # Perform conversion
        logger.info("Starting conversion process...")
        converter.convert()

        logger.info("Conversion completed successfully!")
        logger.info(f"ONNX model saved to: {args.onnx_path}")
        logger.info(f"Config file saved to: {args.config_path}")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

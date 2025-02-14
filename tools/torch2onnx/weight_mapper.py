"""
    Maps weights from old Sequential ConvBlock3D implementation to new individual layer implementation
    
    Old structure:
    conv_block_3d.0.weight  -> conv.weight
    conv_block_3d.1.weight  -> not used (Tanh has no weights)
    conv_block_3d.2.weight  -> norm.weight
    conv_block_3d.2.bias    -> norm.bias
    conv_block_3d.2.running_mean -> not used
    conv_block_3d.2.running_var  -> not used
"""

import torch
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightMapper:
    @staticmethod
    def map_weights(old_state_dict):
        """Maps weights from old Sequential ConvBlock3D to new implementation"""
        new_state_dict = OrderedDict()

        for key, value in old_state_dict.items():
            # Handle module prefix
            if key.startswith('module.'):
                key = key[7:]  # remove 'module.' prefix

            # Skip unnecessary weights
            if any(skip in key for skip in ['fsam', 'bias1']):
                logger.info(f"Skipping unnecessary weight: {key}")
                continue

            # Map ConvBlock3D weights
            if 'conv_block_3d' in key:
                new_key = WeightMapper._map_conv_block_key(key)
                if new_key:
                    new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        return new_state_dict

    @staticmethod
    def _map_conv_block_key(key):
        """Maps individual ConvBlock3D keys"""
        parts = key.split('.')
        layer_idx = int(parts[-2])
        param_name = parts[-1]
        base_name = '.'.join(parts[:-3]) if len(parts) > 3 else ''

        # Map based on layer index
        if layer_idx == 0:  # Conv layer
            component = 'conv'
        # InstanceNorm layer
        elif layer_idx == 2 and param_name in ['weight', 'bias']:
            component = 'norm'
        else:
            return None  # Skip Tanh layer and running stats

        return f"{base_name}.{component}.{param_name}" if base_name else f"{component}.{param_name}"

    @staticmethod
    def verify_mapping(new_model, old_state_dict, new_state_dict):
        """Verifies weight mapping correctness"""
        logger.info("\nVerifying weight mapping...")

        # Check model parameters
        for name, param in new_model.named_parameters():
            if name not in new_state_dict:
                logger.warning(f"Missing weight: {name}")
            elif param.shape != new_state_dict[name].shape:
                logger.error(
                    f"Shape mismatch for {name}: expected {param.shape}, got {new_state_dict[name].shape}")
            else:
                logger.info(f"✓ {name}: Shape {param.shape}")

        # Check for unused weights
        original_keys = set(old_state_dict.keys())
        mapped_keys = set(new_state_dict.keys())
        unused = original_keys - mapped_keys
        if unused:
            logger.info("\nUnused weights from original state dict:")
            for key in unused:
                logger.info(f"- {key}")


def convert_weights(model_path, new_model):
    """Converts weights from old format to new format"""
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')

        # Map weights
        mapper = WeightMapper()
        new_state_dict = mapper.map_weights(state_dict)

        # Verify mapping
        mapper.verify_mapping(new_model, state_dict, new_state_dict)

        # Load weights into model
        missing, unexpected = new_model.load_state_dict(
            new_state_dict, strict=False)

        # Report results
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        return new_state_dict

    except Exception as e:
        logger.error(f"Error converting weights: {str(e)}")
        raise


if __name__ == "__main__":
    # Test weight conversion
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    from MMRPhysSEF import MMRPhysSEF
    model = MMRPhysSEF(frames=300, md_config={
                       "TASKS": ["BVP", "RSP"], "FS": 30}, in_channels=3)
    new_state_dict = convert_weights(args.model_path, model)
    torch.save(new_state_dict, args.save_path)

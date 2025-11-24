#!/usr/bin/env python3
"""
Texture Utilities and PBR Material Classification
==================================================
Handles DDS decompression, texture channel classification, and PBR material
reconstruction from extracted game textures.

Usage:
    python 08_texture_utils.py --input textures/ --output processed/
    python 08_texture_utils.py --classify textures/diffuse_01.dds

Features:
    - DDS BC1-BC7 decompression
    - PBR channel auto-classification (albedo, normal, roughness, metallic)
    - Packed texture channel extraction (ORM maps)
    - Normal map space detection (tangent vs world)
    - Material graph reconstruction

Dependencies:
    pip install numpy pillow

Optional (for advanced DDS formats):
    pip install texture2ddecoder

Author: Capture Pipeline
"""

import argparse
import json
import os
import re
import struct
import sys
from enum import Enum
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("ERROR: numpy and pillow required")
    print("Install: pip install numpy pillow")
    sys.exit(1)

# Optional: texture2ddecoder for BC format decompression
try:
    import texture2ddecoder
    HAS_DECODER = True
except ImportError:
    HAS_DECODER = False


class TextureChannel(Enum):
    """PBR texture channel types."""
    UNKNOWN = "unknown"
    ALBEDO = "albedo"  # Base color (sRGB)
    NORMAL = "normal"  # Tangent-space normals (linear)
    ROUGHNESS = "roughness"  # Roughness (linear)
    METALLIC = "metallic"  # Metallic mask (linear)
    AO = "ao"  # Ambient occlusion (linear)
    EMISSIVE = "emissive"  # Self-illumination (sRGB/HDR)
    ORM = "orm"  # Packed: Occlusion, Roughness, Metallic
    HEIGHT = "height"  # Heightmap/displacement
    OPACITY = "opacity"  # Alpha/transparency


# Filename patterns for channel classification
CHANNEL_PATTERNS = {
    TextureChannel.ALBEDO: [
        r'_?albedo', r'_?basecolor', r'_?diffuse', r'_?color',
        r'_BC$', r'_D$', r'_C$', r'_Base'
    ],
    TextureChannel.NORMAL: [
        r'_?normal', r'_?norm', r'_?nrm', r'_N$', r'_Normal'
    ],
    TextureChannel.ROUGHNESS: [
        r'_?rough', r'_?roughness', r'_R$', r'_Roughness'
    ],
    TextureChannel.METALLIC: [
        r'_?metal', r'_?metallic', r'_?metalness', r'_M$', r'_Metallic'
    ],
    TextureChannel.AO: [
        r'_?ao', r'_?ambient', r'_?occlusion', r'_AO$'
    ],
    TextureChannel.EMISSIVE: [
        r'_?emissive', r'_?emission', r'_?glow', r'_E$'
    ],
    TextureChannel.ORM: [
        r'_?orm', r'_?rma', r'_?arm', r'_ORM$', r'_RMA$'
    ],
    TextureChannel.HEIGHT: [
        r'_?height', r'_?disp', r'_?displacement', r'_H$'
    ],
    TextureChannel.OPACITY: [
        r'_?opacity', r'_?alpha', r'_?mask', r'_O$'
    ]
}


class DDSHeader:
    """DDS file header parser."""

    # DDS magic number
    MAGIC = b'DDS '

    # DXGI formats for DX10 extended header
    DXGI_FORMAT_BC1_UNORM = 71
    DXGI_FORMAT_BC1_UNORM_SRGB = 72
    DXGI_FORMAT_BC2_UNORM = 74
    DXGI_FORMAT_BC3_UNORM = 77
    DXGI_FORMAT_BC4_UNORM = 80
    DXGI_FORMAT_BC5_UNORM = 83
    DXGI_FORMAT_BC6H_UF16 = 95
    DXGI_FORMAT_BC7_UNORM = 98
    DXGI_FORMAT_BC7_UNORM_SRGB = 99

    def __init__(self, filepath):
        self.filepath = filepath
        self.width = 0
        self.height = 0
        self.mipmap_count = 1
        self.format_name = "UNKNOWN"
        self.fourcc = None
        self.has_dx10_header = False
        self.dxgi_format = 0
        self._parse()

    def _parse(self):
        with open(self.filepath, 'rb') as f:
            # Magic
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(f"Not a DDS file: {self.filepath}")

            # DDS_HEADER (124 bytes)
            header = f.read(124)

            self.height = struct.unpack_from('<I', header, 8)[0]
            self.width = struct.unpack_from('<I', header, 12)[0]
            self.mipmap_count = struct.unpack_from('<I', header, 24)[0] or 1

            # Pixel format (offset 72 in header)
            pf_flags = struct.unpack_from('<I', header, 76)[0]
            self.fourcc = header[80:84]

            # Check for DX10 extended header
            if self.fourcc == b'DX10':
                self.has_dx10_header = True
                dx10 = f.read(20)
                self.dxgi_format = struct.unpack_from('<I', dx10, 0)[0]
                self._set_format_from_dxgi()
            else:
                self._set_format_from_fourcc()

    def _set_format_from_fourcc(self):
        fourcc_map = {
            b'DXT1': 'BC1',
            b'DXT3': 'BC2',
            b'DXT5': 'BC3',
            b'ATI1': 'BC4',
            b'ATI2': 'BC5',
            b'BC4U': 'BC4',
            b'BC5U': 'BC5',
        }
        self.format_name = fourcc_map.get(self.fourcc, self.fourcc.decode('ascii', errors='replace'))

    def _set_format_from_dxgi(self):
        dxgi_map = {
            71: 'BC1', 72: 'BC1_SRGB',
            74: 'BC2', 75: 'BC2_SRGB',
            77: 'BC3', 78: 'BC3_SRGB',
            80: 'BC4', 81: 'BC4_SNORM',
            83: 'BC5', 84: 'BC5_SNORM',
            95: 'BC6H_UF16', 96: 'BC6H_SF16',
            98: 'BC7', 99: 'BC7_SRGB',
        }
        self.format_name = dxgi_map.get(self.dxgi_format, f'DXGI_{self.dxgi_format}')


def classify_texture_by_name(filename):
    """
    Classify texture channel by filename patterns.

    Standard conventions:
        - albedo/basecolor/diffuse → ALBEDO
        - normal/norm/nrm → NORMAL
        - rough/roughness → ROUGHNESS
        - metal/metallic → METALLIC
        - ao/occlusion → AO
        - orm/rma → packed ORM
    """
    name_lower = filename.lower()

    for channel, patterns in CHANNEL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return channel

    return TextureChannel.UNKNOWN


def classify_texture_by_content(image_array):
    """
    Classify texture channel by analyzing pixel statistics.

    Heuristics:
        - Normal maps: R/G channels centered around 0.5, B near 1.0
        - Metallic: Bimodal distribution (mostly 0 or 1)
        - Roughness: Broad mid-range values
        - Albedo: Full color range with variation
    """
    if len(image_array.shape) < 3 or image_array.shape[2] < 3:
        return TextureChannel.UNKNOWN

    # Normalize to 0-1
    if image_array.dtype == np.uint8:
        arr = image_array.astype(np.float32) / 255.0
    else:
        arr = image_array.astype(np.float32)

    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    r_std = arr[:, :, 0].std()
    g_std = arr[:, :, 1].std()
    b_std = arr[:, :, 2].std()

    # Normal map detection: R/G centered at 0.5, B near 1.0
    if (0.4 < r_mean < 0.6 and 0.4 < g_mean < 0.6 and b_mean > 0.7):
        return TextureChannel.NORMAL

    # Grayscale detection (roughness/metallic/AO)
    if abs(r_mean - g_mean) < 0.05 and abs(g_mean - b_mean) < 0.05:
        # Check distribution
        flat = arr[:, :, 0].flatten()
        hist, _ = np.histogram(flat, bins=16, range=(0, 1))

        # Bimodal (mostly 0 or 1) = metallic
        if hist[0] + hist[-1] > 0.7 * len(flat):
            return TextureChannel.METALLIC

        # Mid-range peak = roughness
        mid_sum = hist[4:12].sum()
        if mid_sum > 0.5 * len(flat):
            return TextureChannel.ROUGHNESS

        return TextureChannel.AO

    return TextureChannel.ALBEDO


def decompress_dds(filepath, output_path=None):
    """
    Decompress DDS texture to PNG.

    Supports BC1-BC7 formats via texture2ddecoder if available,
    otherwise falls back to pillow-supported formats.
    """
    try:
        # Try pillow first (handles uncompressed and some DXT)
        img = Image.open(filepath)
        img_array = np.array(img)

        if output_path:
            img.save(output_path)

        return img_array, img.mode

    except Exception:
        pass

    # Try texture2ddecoder for BC formats
    if HAS_DECODER:
        try:
            header = DDSHeader(filepath)

            with open(filepath, 'rb') as f:
                # Skip headers
                f.seek(128)
                if header.has_dx10_header:
                    f.seek(148)
                data = f.read()

            # Decode based on format
            if 'BC1' in header.format_name:
                decoded = texture2ddecoder.decode_bc1(data, header.width, header.height)
            elif 'BC3' in header.format_name:
                decoded = texture2ddecoder.decode_bc3(data, header.width, header.height)
            elif 'BC4' in header.format_name:
                decoded = texture2ddecoder.decode_bc4(data, header.width, header.height)
            elif 'BC5' in header.format_name:
                decoded = texture2ddecoder.decode_bc5(data, header.width, header.height)
            elif 'BC7' in header.format_name:
                decoded = texture2ddecoder.decode_bc7(data, header.width, header.height)
            else:
                raise ValueError(f"Unsupported format: {header.format_name}")

            # Convert to numpy array
            img_array = np.frombuffer(decoded, dtype=np.uint8)
            img_array = img_array.reshape((header.height, header.width, 4))

            if output_path:
                img = Image.fromarray(img_array)
                img.save(output_path)

            return img_array, 'RGBA'

        except Exception as e:
            print(f"  Warning: DDS decode failed for {filepath}: {e}")

    return None, None


def extract_orm_channels(orm_image, output_dir, base_name):
    """
    Extract individual channels from packed ORM texture.

    ORM packing:
        R = Ambient Occlusion
        G = Roughness
        B = Metallic
    """
    if orm_image is None:
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Ensure RGB
    if len(orm_image.shape) < 3:
        return results

    # AO (R channel)
    ao = orm_image[:, :, 0]
    ao_path = output_path / f"{base_name}_AO.png"
    Image.fromarray(ao).save(ao_path)
    results['ao'] = str(ao_path)

    # Roughness (G channel)
    roughness = orm_image[:, :, 1]
    rough_path = output_path / f"{base_name}_Roughness.png"
    Image.fromarray(roughness).save(rough_path)
    results['roughness'] = str(rough_path)

    # Metallic (B channel)
    metallic = orm_image[:, :, 2]
    metal_path = output_path / f"{base_name}_Metallic.png"
    Image.fromarray(metallic).save(metal_path)
    results['metallic'] = str(metal_path)

    return results


def process_texture_directory(input_dir, output_dir):
    """
    Process all textures in directory: classify, decompress, organize.

    Returns material database grouping textures by material name.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return {}

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all texture files
    extensions = ['*.dds', '*.png', '*.jpg', '*.jpeg', '*.tga', '*.bmp', '*.exr']
    texture_files = []
    for ext in extensions:
        texture_files.extend(input_path.glob(f"**/{ext}"))

    print(f"Found {len(texture_files)} texture files")

    # Process and classify
    materials = {}  # material_name -> {channel: path}

    for tex_path in texture_files:
        try:
            filename = tex_path.stem
            ext = tex_path.suffix.lower()

            # Classify by name
            channel = classify_texture_by_name(filename)

            # Decompress DDS if needed
            if ext == '.dds':
                out_png = output_path / f"{filename}.png"
                img_array, mode = decompress_dds(str(tex_path), str(out_png))

                if img_array is None:
                    continue

                # Classify by content if unknown
                if channel == TextureChannel.UNKNOWN:
                    channel = classify_texture_by_content(img_array)

                tex_output = str(out_png)

                # Handle packed ORM
                if channel == TextureChannel.ORM:
                    extract_orm_channels(img_array, str(output_path), filename)

            else:
                # Copy non-DDS files
                img = Image.open(tex_path)
                img_array = np.array(img)

                if channel == TextureChannel.UNKNOWN:
                    channel = classify_texture_by_content(img_array)

                out_file = output_path / f"{filename}{ext}"
                img.save(out_file)
                tex_output = str(out_file)

            # Group by material name
            # Extract material name by removing channel suffix
            mat_name = re.sub(r'_(albedo|normal|roughness|metallic|ao|orm|emissive|height).*$',
                              '', filename, flags=re.IGNORECASE)

            if mat_name not in materials:
                materials[mat_name] = {}

            materials[mat_name][channel.value] = tex_output

            print(f"  {filename}: {channel.value}")

        except Exception as e:
            print(f"  Error processing {tex_path}: {e}")

    return materials


def generate_material_json(materials, output_path):
    """
    Generate material definition JSON for use in Blender/engine import.
    """
    material_defs = []

    for mat_name, channels in materials.items():
        mat_def = {
            "name": mat_name,
            "workflow": "metallic_roughness",
            "channels": channels,
            "color_space": {
                "albedo": "sRGB",
                "emissive": "sRGB",
                "normal": "Linear",
                "roughness": "Linear",
                "metallic": "Linear",
                "ao": "Linear"
            }
        }
        material_defs.append(mat_def)

    with open(output_path, 'w') as f:
        json.dump({
            "version": "1.0",
            "materials": material_defs
        }, f, indent=2)

    return material_defs


def main():
    parser = argparse.ArgumentParser(
        description="Texture utilities and PBR material classification"
    )
    parser.add_argument(
        '--input', '-i',
        help="Input directory containing textures"
    )
    parser.add_argument(
        '--output', '-o',
        default='processed_textures',
        help="Output directory"
    )
    parser.add_argument(
        '--classify',
        help="Classify a single texture file"
    )

    args = parser.parse_args()

    if args.classify:
        # Single file classification
        filepath = args.classify
        print(f"\n=== Texture Classification ===")
        print(f"File: {filepath}")

        # By name
        channel = classify_texture_by_name(Path(filepath).name)
        print(f"By filename: {channel.value}")

        # By content
        img_array, mode = decompress_dds(filepath) if filepath.lower().endswith('.dds') else (
            np.array(Image.open(filepath)), 'RGB'
        )

        if img_array is not None:
            channel = classify_texture_by_content(img_array)
            print(f"By content: {channel.value}")

        return

    if args.input:
        print(f"\n=== Texture Processing ===")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")

        materials = process_texture_directory(args.input, args.output)

        # Generate material JSON
        mat_json = Path(args.output) / "materials.json"
        generate_material_json(materials, str(mat_json))
        print(f"\nMaterial definitions: {mat_json}")

        print(f"\nProcessed {len(materials)} material groups")
        print("\n[OK] Texture processing complete!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

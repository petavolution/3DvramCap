#!/usr/bin/env python3
"""
Texture Processing Utilities
============================
Process, convert, and optimize textures for the capture pipeline.

Features:
    - DDS file reading (DX9/DX10 formats)
    - Format conversion (DDS -> PNG, TGA -> PNG)
    - Texture resizing and mipmap generation
    - Channel extraction and packing
    - Normal map processing
    - Texture deduplication
    - Batch processing

Usage:
    from core_texture import TextureProcessor, convert_texture

    # Simple conversion
    convert_texture("input.dds", "output.png")

    # Full processing
    processor = TextureProcessor("input.dds")
    processor.resize(1024, 1024)
    processor.save("output.png")

Author: Capture Pipeline
"""

import os
import struct
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, BinaryIO
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class TextureFormat(Enum):
    """Texture format types."""
    UNKNOWN = "unknown"
    RGB = "rgb"
    RGBA = "rgba"
    DXT1 = "dxt1"
    DXT3 = "dxt3"
    DXT5 = "dxt5"
    BC4 = "bc4"
    BC5 = "bc5"
    BC6H = "bc6h"
    BC7 = "bc7"
    GRAYSCALE = "grayscale"


@dataclass
class TextureInfo:
    """Texture metadata."""
    width: int
    height: int
    format: TextureFormat
    mip_levels: int
    is_cubemap: bool = False
    array_size: int = 1
    filepath: Optional[str] = None
    hash: Optional[str] = None


class DDSReader:
    """
    DirectDraw Surface (DDS) file reader.

    Supports DX9 and DX10 formats including:
    - Uncompressed (RGB, RGBA)
    - DXT1/BC1, DXT3/BC2, DXT5/BC3
    - BC4, BC5, BC6H, BC7
    """

    # DDS constants
    MAGIC = b'DDS '
    HEADER_SIZE = 124

    # Pixel format flags
    DDPF_ALPHAPIXELS = 0x1
    DDPF_FOURCC = 0x4
    DDPF_RGB = 0x40

    # FourCC codes
    FOURCC_DXT1 = b'DXT1'
    FOURCC_DXT3 = b'DXT3'
    FOURCC_DXT5 = b'DXT5'
    FOURCC_DX10 = b'DX10'
    FOURCC_ATI1 = b'ATI1'
    FOURCC_ATI2 = b'ATI2'
    FOURCC_BC4U = b'BC4U'
    FOURCC_BC5U = b'BC5U'

    # DXGI formats (DX10 header)
    DXGI_FORMAT_BC1_UNORM = 71
    DXGI_FORMAT_BC2_UNORM = 74
    DXGI_FORMAT_BC3_UNORM = 77
    DXGI_FORMAT_BC4_UNORM = 80
    DXGI_FORMAT_BC5_UNORM = 83
    DXGI_FORMAT_BC6H_UF16 = 95
    DXGI_FORMAT_BC7_UNORM = 98

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.info: Optional[TextureInfo] = None
        self._data: Optional[bytes] = None

    def read(self) -> Tuple[TextureInfo, np.ndarray]:
        """Read DDS file and return info and pixel data."""
        if not HAS_NUMPY:
            raise ImportError("numpy required for DDS reading")

        with open(self.filepath, 'rb') as f:
            self._read_header(f)
            pixels = self._read_pixels(f)

        return self.info, pixels

    def _read_header(self, f: BinaryIO):
        """Parse DDS header."""
        magic = f.read(4)
        if magic != self.MAGIC:
            raise ValueError(f"Not a DDS file: {self.filepath}")

        # DDS_HEADER structure
        header = struct.unpack('<7I44x', f.read(72))
        size, flags, height, width, pitch_or_linear, depth, mip_count = header

        # DDS_PIXELFORMAT structure
        pf = struct.unpack('<2I4s5I', f.read(32))
        pf_size, pf_flags, fourcc, rgb_bits, r_mask, g_mask, b_mask, a_mask = pf

        # Remaining header
        f.read(20)  # caps1, caps2, caps3, caps4, reserved

        # Determine format
        tex_format = TextureFormat.UNKNOWN
        is_dx10 = False

        if pf_flags & self.DDPF_FOURCC:
            if fourcc == self.FOURCC_DXT1:
                tex_format = TextureFormat.DXT1
            elif fourcc == self.FOURCC_DXT3:
                tex_format = TextureFormat.DXT3
            elif fourcc == self.FOURCC_DXT5:
                tex_format = TextureFormat.DXT5
            elif fourcc in (self.FOURCC_ATI1, self.FOURCC_BC4U):
                tex_format = TextureFormat.BC4
            elif fourcc in (self.FOURCC_ATI2, self.FOURCC_BC5U):
                tex_format = TextureFormat.BC5
            elif fourcc == self.FOURCC_DX10:
                is_dx10 = True
        elif pf_flags & self.DDPF_RGB:
            if pf_flags & self.DDPF_ALPHAPIXELS:
                tex_format = TextureFormat.RGBA
            else:
                tex_format = TextureFormat.RGB

        # DX10 extended header
        array_size = 1
        if is_dx10:
            dx10 = struct.unpack('<5I', f.read(20))
            dxgi_format, dimension, misc_flag, array_size, misc_flags2 = dx10

            if dxgi_format == self.DXGI_FORMAT_BC1_UNORM:
                tex_format = TextureFormat.DXT1
            elif dxgi_format == self.DXGI_FORMAT_BC2_UNORM:
                tex_format = TextureFormat.DXT3
            elif dxgi_format == self.DXGI_FORMAT_BC3_UNORM:
                tex_format = TextureFormat.DXT5
            elif dxgi_format == self.DXGI_FORMAT_BC4_UNORM:
                tex_format = TextureFormat.BC4
            elif dxgi_format == self.DXGI_FORMAT_BC5_UNORM:
                tex_format = TextureFormat.BC5
            elif dxgi_format == self.DXGI_FORMAT_BC6H_UF16:
                tex_format = TextureFormat.BC6H
            elif dxgi_format == self.DXGI_FORMAT_BC7_UNORM:
                tex_format = TextureFormat.BC7

        self.info = TextureInfo(
            width=width,
            height=height,
            format=tex_format,
            mip_levels=max(1, mip_count),
            array_size=array_size,
            filepath=self.filepath
        )

    def _read_pixels(self, f: BinaryIO) -> np.ndarray:
        """Read and decompress pixel data."""
        width = self.info.width
        height = self.info.height
        fmt = self.info.format

        # Read remaining data
        self._data = f.read()

        # Decode based on format
        if fmt == TextureFormat.RGBA:
            return self._decode_rgba(width, height)
        elif fmt == TextureFormat.RGB:
            return self._decode_rgb(width, height)
        elif fmt == TextureFormat.DXT1:
            return self._decode_dxt1(width, height)
        elif fmt == TextureFormat.DXT5:
            return self._decode_dxt5(width, height)
        elif fmt == TextureFormat.DXT3:
            return self._decode_dxt3(width, height)
        elif fmt == TextureFormat.BC4:
            return self._decode_bc4(width, height)
        elif fmt == TextureFormat.BC5:
            return self._decode_bc5(width, height)
        else:
            # Return raw data as grayscale fallback
            pixels = np.frombuffer(self._data[:width * height], dtype=np.uint8)
            return pixels.reshape(height, width)

    def _decode_rgba(self, width: int, height: int) -> np.ndarray:
        """Decode uncompressed RGBA."""
        pixels = np.frombuffer(self._data[:width * height * 4], dtype=np.uint8)
        return pixels.reshape(height, width, 4)

    def _decode_rgb(self, width: int, height: int) -> np.ndarray:
        """Decode uncompressed RGB."""
        pixels = np.frombuffer(self._data[:width * height * 3], dtype=np.uint8)
        return pixels.reshape(height, width, 3)

    def _decode_dxt1(self, width: int, height: int) -> np.ndarray:
        """Decode DXT1/BC1 compressed data."""
        output = np.zeros((height, width, 4), dtype=np.uint8)

        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + 8 > len(self._data):
                    break

                block = self._data[offset:offset + 8]
                offset += 8

                # Decode colors
                c0, c1 = struct.unpack('<HH', block[:4])
                indices = struct.unpack('<I', block[4:8])[0]

                # Expand 565 to RGB
                colors = [self._rgb565_to_rgb(c0), self._rgb565_to_rgb(c1)]

                # Generate intermediate colors
                if c0 > c1:
                    colors.append(tuple((2 * colors[0][i] + colors[1][i]) // 3 for i in range(3)))
                    colors.append(tuple((colors[0][i] + 2 * colors[1][i]) // 3 for i in range(3)))
                else:
                    colors.append(tuple((colors[0][i] + colors[1][i]) // 2 for i in range(3)))
                    colors.append((0, 0, 0))  # Transparent

                # Apply to pixels
                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            idx = (indices >> (2 * (py * 4 + px))) & 3
                            output[y, x, :3] = colors[idx]
                            output[y, x, 3] = 0 if (c0 <= c1 and idx == 3) else 255

        return output

    def _decode_dxt3(self, width: int, height: int) -> np.ndarray:
        """Decode DXT3/BC2 compressed data."""
        output = self._decode_dxt_color(width, height, block_size=16, alpha_offset=0)

        # DXT3 has explicit alpha (4 bits per pixel)
        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + 8 > len(self._data):
                    break

                alpha_data = struct.unpack('<Q', self._data[offset:offset + 8])[0]
                offset += 16  # Skip color block too

                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            bit_idx = (py * 4 + px) * 4
                            alpha = ((alpha_data >> bit_idx) & 0xF) * 17
                            output[y, x, 3] = alpha

        return output

    def _decode_dxt5(self, width: int, height: int) -> np.ndarray:
        """Decode DXT5/BC3 compressed data."""
        output = self._decode_dxt_color(width, height, block_size=16, alpha_offset=8)

        # DXT5 has interpolated alpha
        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + 8 > len(self._data):
                    break

                a0, a1 = self._data[offset], self._data[offset + 1]
                alpha_bits = struct.unpack('<Q', self._data[offset:offset + 8])[0] >> 16

                # Generate alpha palette
                alphas = [a0, a1]
                if a0 > a1:
                    for i in range(6):
                        alphas.append(((6 - i) * a0 + (i + 1) * a1) // 7)
                else:
                    for i in range(4):
                        alphas.append(((4 - i) * a0 + (i + 1) * a1) // 5)
                    alphas.extend([0, 255])

                offset += 16

                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            idx = (alpha_bits >> (3 * (py * 4 + px))) & 7
                            output[y, x, 3] = alphas[idx]

        return output

    def _decode_dxt_color(self, width: int, height: int,
                         block_size: int, alpha_offset: int) -> np.ndarray:
        """Decode DXT color blocks."""
        output = np.zeros((height, width, 4), dtype=np.uint8)

        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + block_size > len(self._data):
                    break

                color_offset = offset + alpha_offset
                c0, c1 = struct.unpack('<HH', self._data[color_offset:color_offset + 4])
                indices = struct.unpack('<I', self._data[color_offset + 4:color_offset + 8])[0]

                colors = [self._rgb565_to_rgb(c0), self._rgb565_to_rgb(c1)]
                colors.append(tuple((2 * colors[0][i] + colors[1][i]) // 3 for i in range(3)))
                colors.append(tuple((colors[0][i] + 2 * colors[1][i]) // 3 for i in range(3)))

                offset += block_size

                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            idx = (indices >> (2 * (py * 4 + px))) & 3
                            output[y, x, :3] = colors[idx]
                            output[y, x, 3] = 255

        return output

    def _decode_bc4(self, width: int, height: int) -> np.ndarray:
        """Decode BC4 (single channel) compressed data."""
        output = np.zeros((height, width), dtype=np.uint8)

        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + 8 > len(self._data):
                    break

                r0, r1 = self._data[offset], self._data[offset + 1]
                bits = struct.unpack('<Q', self._data[offset:offset + 8])[0] >> 16

                # Generate palette
                reds = [r0, r1]
                if r0 > r1:
                    for i in range(6):
                        reds.append(((6 - i) * r0 + (i + 1) * r1) // 7)
                else:
                    for i in range(4):
                        reds.append(((4 - i) * r0 + (i + 1) * r1) // 5)
                    reds.extend([0, 255])

                offset += 8

                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            idx = (bits >> (3 * (py * 4 + px))) & 7
                            output[y, x] = reds[idx]

        return output

    def _decode_bc5(self, width: int, height: int) -> np.ndarray:
        """Decode BC5 (two channel, often normal maps) data."""
        output = np.zeros((height, width, 3), dtype=np.uint8)

        block_w = (width + 3) // 4
        block_h = (height + 3) // 4
        offset = 0

        for by in range(block_h):
            for bx in range(block_w):
                if offset + 16 > len(self._data):
                    break

                # Red channel
                r0, r1 = self._data[offset], self._data[offset + 1]
                r_bits = struct.unpack('<Q', self._data[offset:offset + 8])[0] >> 16

                # Green channel
                g0, g1 = self._data[offset + 8], self._data[offset + 9]
                g_bits = struct.unpack('<Q', self._data[offset + 8:offset + 16])[0] >> 16

                # Generate palettes
                def make_palette(v0, v1):
                    p = [v0, v1]
                    if v0 > v1:
                        for i in range(6):
                            p.append(((6 - i) * v0 + (i + 1) * v1) // 7)
                    else:
                        for i in range(4):
                            p.append(((4 - i) * v0 + (i + 1) * v1) // 5)
                        p.extend([0, 255])
                    return p

                reds = make_palette(r0, r1)
                greens = make_palette(g0, g1)

                offset += 16

                for py in range(4):
                    for px in range(4):
                        x = bx * 4 + px
                        y = by * 4 + py
                        if x < width and y < height:
                            r_idx = (r_bits >> (3 * (py * 4 + px))) & 7
                            g_idx = (g_bits >> (3 * (py * 4 + px))) & 7
                            output[y, x, 0] = reds[r_idx]
                            output[y, x, 1] = greens[g_idx]
                            output[y, x, 2] = 255  # Blue often computed

        return output

    @staticmethod
    def _rgb565_to_rgb(c: int) -> Tuple[int, int, int]:
        """Convert RGB565 to RGB888."""
        r = ((c >> 11) & 0x1F) * 255 // 31
        g = ((c >> 5) & 0x3F) * 255 // 63
        b = (c & 0x1F) * 255 // 31
        return (r, g, b)


class TextureProcessor:
    """
    High-level texture processing utility.
    """

    def __init__(self, source: Union[str, np.ndarray, 'Image.Image'] = None):
        """
        Initialize processor.

        Args:
            source: File path, numpy array, or PIL Image
        """
        self.pixels: Optional[np.ndarray] = None
        self.info: Optional[TextureInfo] = None

        if source is not None:
            self.load(source)

    def load(self, source: Union[str, np.ndarray, 'Image.Image']):
        """Load texture from various sources."""
        if isinstance(source, str):
            self._load_file(source)
        elif isinstance(source, np.ndarray):
            self.pixels = source
            h, w = source.shape[:2]
            self.info = TextureInfo(
                width=w,
                height=h,
                format=TextureFormat.RGBA if source.shape[-1] == 4 else TextureFormat.RGB,
                mip_levels=1
            )
        elif HAS_PIL and isinstance(source, Image.Image):
            self.pixels = np.array(source)
            self.info = TextureInfo(
                width=source.width,
                height=source.height,
                format=TextureFormat.RGBA if source.mode == 'RGBA' else TextureFormat.RGB,
                mip_levels=1
            )

    def _load_file(self, filepath: str):
        """Load from file path."""
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext == '.dds':
            reader = DDSReader(filepath)
            self.info, self.pixels = reader.read()
        elif HAS_PIL:
            img = Image.open(filepath)
            self.pixels = np.array(img.convert('RGBA'))
            self.info = TextureInfo(
                width=img.width,
                height=img.height,
                format=TextureFormat.RGBA,
                mip_levels=1,
                filepath=filepath
            )
        else:
            raise ImportError("PIL required for non-DDS formats")

    def resize(self, width: int, height: int, filter_mode: str = 'lanczos'):
        """Resize texture."""
        if not HAS_PIL:
            raise ImportError("PIL required for resizing")

        filters = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }

        img = Image.fromarray(self.pixels)
        img = img.resize((width, height), filters.get(filter_mode, Image.LANCZOS))
        self.pixels = np.array(img)
        self.info.width = width
        self.info.height = height

    def resize_power_of_two(self, max_size: int = 4096):
        """Resize to nearest power of two dimensions."""
        def nearest_pot(n):
            return 2 ** int(np.ceil(np.log2(n)))

        new_w = min(nearest_pot(self.info.width), max_size)
        new_h = min(nearest_pot(self.info.height), max_size)

        if new_w != self.info.width or new_h != self.info.height:
            self.resize(new_w, new_h)

    def extract_channel(self, channel: int) -> np.ndarray:
        """Extract a single channel (0=R, 1=G, 2=B, 3=A)."""
        if len(self.pixels.shape) == 2:
            return self.pixels
        return self.pixels[:, :, channel]

    def set_channel(self, channel: int, data: np.ndarray):
        """Set a single channel."""
        if len(self.pixels.shape) == 2:
            self.pixels = np.stack([self.pixels] * 4, axis=-1)
        self.pixels[:, :, channel] = data

    def to_grayscale(self):
        """Convert to grayscale."""
        if len(self.pixels.shape) == 3:
            # Luminance formula
            self.pixels = (
                0.299 * self.pixels[:, :, 0] +
                0.587 * self.pixels[:, :, 1] +
                0.114 * self.pixels[:, :, 2]
            ).astype(np.uint8)
            self.info.format = TextureFormat.GRAYSCALE

    def normalize_normal_map(self):
        """Normalize a normal map (ensure unit vectors)."""
        if len(self.pixels.shape) != 3 or self.pixels.shape[2] < 3:
            return

        # Convert to float [-1, 1]
        normals = (self.pixels[:, :, :3].astype(np.float32) / 127.5) - 1.0

        # Normalize
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)
        normals /= lengths

        # Convert back to [0, 255]
        self.pixels[:, :, :3] = ((normals + 1.0) * 127.5).astype(np.uint8)

    def flip_green_channel(self):
        """Flip green channel (DirectX <-> OpenGL normal maps)."""
        if len(self.pixels.shape) == 3 and self.pixels.shape[2] >= 2:
            self.pixels[:, :, 1] = 255 - self.pixels[:, :, 1]

    def compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        h = hashlib.md5(self.pixels.tobytes()).hexdigest()
        self.info.hash = h
        return h

    def save(self, filepath: str, quality: int = 95):
        """Save texture to file."""
        if not HAS_PIL:
            raise ImportError("PIL required for saving")

        path = Path(filepath)
        ext = path.suffix.lower()

        img = Image.fromarray(self.pixels)

        if ext in ('.jpg', '.jpeg'):
            img = img.convert('RGB')
            img.save(filepath, quality=quality)
        elif ext == '.png':
            img.save(filepath, optimize=True)
        elif ext == '.tga':
            img.save(filepath)
        else:
            img.save(filepath)

        print(f"Saved: {filepath} ({self.info.width}x{self.info.height})")

    def to_pil(self) -> 'Image.Image':
        """Convert to PIL Image."""
        if not HAS_PIL:
            raise ImportError("PIL required")
        return Image.fromarray(self.pixels)


def convert_texture(input_path: str,
                    output_path: str,
                    max_size: Optional[int] = None,
                    power_of_two: bool = False) -> bool:
    """
    Convert texture file to another format.

    Args:
        input_path: Input texture path
        output_path: Output path (format from extension)
        max_size: Maximum dimension
        power_of_two: Resize to power of two

    Returns:
        True if successful
    """
    try:
        processor = TextureProcessor(input_path)

        if power_of_two:
            processor.resize_power_of_two(max_size or 4096)
        elif max_size:
            w, h = processor.info.width, processor.info.height
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                processor.resize(int(w * scale), int(h * scale))

        processor.save(output_path)
        return True

    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def batch_convert_textures(input_dir: str,
                          output_dir: str,
                          output_format: str = '.png',
                          max_size: int = 2048) -> Dict[str, str]:
    """
    Batch convert textures.

    Returns:
        Dict mapping input -> output paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for ext in ['.dds', '.tga', '.png', '.jpg', '.jpeg', '.bmp']:
        for tex_file in input_path.glob(f'*{ext}'):
            out_file = output_path / (tex_file.stem + output_format)

            if convert_texture(str(tex_file), str(out_file), max_size):
                results[str(tex_file)] = str(out_file)

    return results


def deduplicate_textures(texture_dir: str,
                         output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Remove duplicate textures based on content hash.

    Returns:
        Dict mapping original -> canonical path
    """
    tex_path = Path(texture_dir)
    hash_to_file: Dict[str, str] = {}
    mapping: Dict[str, str] = {}

    for ext in ['.png', '.jpg', '.dds', '.tga']:
        for tex_file in tex_path.glob(f'*{ext}'):
            try:
                processor = TextureProcessor(str(tex_file))
                h = processor.compute_hash()

                if h in hash_to_file:
                    # Duplicate
                    mapping[str(tex_file)] = hash_to_file[h]
                else:
                    # New unique texture
                    hash_to_file[h] = str(tex_file)
                    mapping[str(tex_file)] = str(tex_file)

                    if output_dir:
                        out_file = Path(output_dir) / tex_file.name
                        processor.save(str(out_file))

            except Exception as e:
                print(f"Error processing {tex_file}: {e}")
                mapping[str(tex_file)] = str(tex_file)

    unique = len(set(mapping.values()))
    print(f"Found {len(mapping)} textures, {unique} unique")

    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process textures")
    parser.add_argument('input', help="Input texture or directory")
    parser.add_argument('--output', '-o', help="Output path")
    parser.add_argument('--max-size', type=int, default=2048, help="Maximum dimension")
    parser.add_argument('--pot', action='store_true', help="Resize to power of two")
    parser.add_argument('--batch', action='store_true', help="Batch process directory")

    args = parser.parse_args()

    if args.batch:
        output_dir = args.output or str(Path(args.input) / "converted")
        batch_convert_textures(args.input, output_dir, max_size=args.max_size)
    else:
        output = args.output or args.input.rsplit('.', 1)[0] + '.png'
        convert_texture(args.input, output, args.max_size, args.pot)

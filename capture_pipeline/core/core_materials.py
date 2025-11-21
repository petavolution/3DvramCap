#!/usr/bin/env python3
"""
Core Material and Texture Extraction
=====================================
Extracts material properties and texture bindings from RenderDoc captures.

Features:
    - Shader resource binding extraction
    - Texture slot identification (albedo, normal, roughness, etc.)
    - Constant buffer parameter extraction
    - Material property reconstruction

Usage (in RenderDoc environment):
    from core_materials import MaterialExtractor

    extractor = MaterialExtractor(controller)
    materials = extractor.extract_draw_call(event_id)

Author: Capture Pipeline
"""

import os
import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import renderdoc as rd
    HAS_RENDERDOC = True
except ImportError:
    HAS_RENDERDOC = False


class TextureSlot(Enum):
    """Common texture slot types in game materials."""
    ALBEDO = "albedo"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    AO = "ao"
    EMISSIVE = "emissive"
    OPACITY = "opacity"
    HEIGHT = "height"
    SPECULAR = "specular"
    ORM = "orm"  # Packed: Occlusion/Roughness/Metallic
    MASK = "mask"
    UNKNOWN = "unknown"


@dataclass
class TextureBinding:
    """A texture bound to a shader slot."""
    resource_id: int
    slot_index: int
    slot_name: str
    texture_type: TextureSlot
    width: int = 0
    height: int = 0
    format: str = ""
    filepath: str = ""  # After extraction
    sampler_info: dict = field(default_factory=dict)


@dataclass
class MaterialParameter:
    """A material parameter from constant buffer."""
    name: str
    value: Any
    type: str  # float, float2, float3, float4, int, etc.


@dataclass
class ExtractedMaterial:
    """Complete extracted material."""
    name: str
    event_id: int
    shader_name: str = ""
    textures: List[TextureBinding] = field(default_factory=list)
    parameters: List[MaterialParameter] = field(default_factory=list)
    is_transparent: bool = False
    is_double_sided: bool = False
    blend_mode: str = "opaque"

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'event_id': self.event_id,
            'shader': self.shader_name,
            'textures': [
                {
                    'slot': t.slot_name,
                    'type': t.texture_type.value,
                    'filepath': t.filepath,
                    'size': [t.width, t.height],
                    'format': t.format
                }
                for t in self.textures
            ],
            'parameters': [
                {'name': p.name, 'value': p.value, 'type': p.type}
                for p in self.parameters
            ],
            'transparent': self.is_transparent,
            'double_sided': self.is_double_sided,
            'blend_mode': self.blend_mode
        }


# Texture slot name patterns for automatic detection
TEXTURE_SLOT_PATTERNS = {
    TextureSlot.ALBEDO: [
        r'albedo', r'diffuse', r'base.?color', r'color', r'_d$', r'_diff',
        r'_albedo', r'_bc$', r'_basecolor'
    ],
    TextureSlot.NORMAL: [
        r'normal', r'bump', r'_n$', r'_norm', r'_nrm', r'normalmap'
    ],
    TextureSlot.ROUGHNESS: [
        r'roughness', r'rough', r'_r$', r'_rough', r'glossiness', r'gloss'
    ],
    TextureSlot.METALLIC: [
        r'metallic', r'metal', r'_m$', r'_met', r'metalness'
    ],
    TextureSlot.AO: [
        r'occlusion', r'_ao$', r'ambient', r'_occ'
    ],
    TextureSlot.EMISSIVE: [
        r'emissive', r'emission', r'_e$', r'_emit', r'glow'
    ],
    TextureSlot.ORM: [
        r'_orm', r'packed', r'_rma', r'_arm', r'occlusionroughnessmetallic'
    ],
    TextureSlot.OPACITY: [
        r'opacity', r'alpha', r'_o$', r'_alpha', r'transparency'
    ],
    TextureSlot.HEIGHT: [
        r'height', r'displacement', r'_h$', r'_disp', r'parallax'
    ],
    TextureSlot.SPECULAR: [
        r'specular', r'_s$', r'_spec', r'reflection'
    ],
    TextureSlot.MASK: [
        r'mask', r'_mask', r'blend'
    ]
}


def classify_texture_slot(name: str) -> TextureSlot:
    """
    Classify a texture slot based on its name.

    Args:
        name: Texture slot/variable name

    Returns:
        Detected TextureSlot type
    """
    name_lower = name.lower()

    for slot_type, patterns in TEXTURE_SLOT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return slot_type

    return TextureSlot.UNKNOWN


class MaterialExtractor:
    """
    Extracts material and texture information from RenderDoc captures.

    Handles:
        - D3D11, D3D12, Vulkan, OpenGL resource bindings
        - Texture slot classification
        - Constant buffer parameter extraction
        - Texture data saving
    """

    def __init__(self, controller, output_dir: str = "export/Materials"):
        """
        Initialize material extractor.

        Args:
            controller: RenderDoc ReplayController
            output_dir: Directory to save extracted textures
        """
        if not HAS_RENDERDOC:
            raise RuntimeError("RenderDoc module not available")

        self.controller = controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for texture info to avoid repeated queries
        self._texture_cache: Dict[int, dict] = {}

    def extract_draw_call(self, event_id: int,
                          save_textures: bool = True) -> Optional[ExtractedMaterial]:
        """
        Extract material information for a specific draw call.

        Args:
            event_id: RenderDoc event ID
            save_textures: Whether to save texture data to files

        Returns:
            ExtractedMaterial or None
        """
        self.controller.SetFrameEvent(event_id, False)

        state = self.controller.GetPipelineState()

        # Get pixel shader info
        ps = state.GetShader(rd.ShaderStage.Pixel)
        if ps == rd.ResourceId.Null():
            return None

        ps_refl = state.GetShaderReflection(rd.ShaderStage.Pixel)

        # Create material
        material = ExtractedMaterial(
            name=f"material_{event_id}",
            event_id=event_id,
            shader_name=self._get_shader_name(ps_refl)
        )

        # Extract texture bindings
        material.textures = self._extract_textures(
            state, ps_refl, rd.ShaderStage.Pixel, save_textures
        )

        # Extract constant buffer parameters
        material.parameters = self._extract_parameters(
            state, ps_refl, rd.ShaderStage.Pixel
        )

        # Detect blend mode
        material.blend_mode = self._detect_blend_mode(state)
        material.is_transparent = material.blend_mode != "opaque"

        return material

    def _get_shader_name(self, reflection) -> str:
        """Get shader name from reflection."""
        if reflection and hasattr(reflection, 'debugInfo'):
            if reflection.debugInfo and hasattr(reflection.debugInfo, 'entryName'):
                return reflection.debugInfo.entryName
        return "unknown"

    def _extract_textures(self, state, reflection, stage: 'rd.ShaderStage',
                          save_textures: bool) -> List[TextureBinding]:
        """Extract texture bindings from shader resources."""
        textures = []

        if not reflection or not reflection.readOnlyResources:
            return textures

        for i, res in enumerate(reflection.readOnlyResources):
            if not res.isTexture:
                continue

            # Get resource binding
            binding = self._get_resource_binding(state, stage, i)
            if binding is None:
                continue

            # Get texture info
            tex_info = self._get_texture_info(binding.resource_id)
            if tex_info is None:
                continue

            # Classify slot type
            slot_type = classify_texture_slot(res.name)

            texture = TextureBinding(
                resource_id=int(binding.resource_id),
                slot_index=i,
                slot_name=res.name,
                texture_type=slot_type,
                width=tex_info.get('width', 0),
                height=tex_info.get('height', 0),
                format=tex_info.get('format', '')
            )

            # Save texture if requested
            if save_textures and tex_info.get('width', 0) > 0:
                filepath = self._save_texture(
                    binding.resource_id,
                    f"{texture.slot_name}_{binding.resource_id}"
                )
                texture.filepath = str(filepath) if filepath else ""

            textures.append(texture)

        return textures

    def _get_resource_binding(self, state, stage, index) -> Optional[Any]:
        """Get resource binding for a shader resource index."""
        try:
            # Use API abstraction to get bindings
            bindings = state.GetReadOnlyResources(stage, True)
            if index < len(bindings):
                bind = bindings[index]
                if bind.resources:
                    return bind.resources[0]
        except Exception:
            pass
        return None

    def _get_texture_info(self, resource_id) -> Optional[dict]:
        """Get texture information from resource ID."""
        rid = int(resource_id)

        if rid in self._texture_cache:
            return self._texture_cache[rid]

        try:
            textures = self.controller.GetTextures()
            for tex in textures:
                if int(tex.resourceId) == rid:
                    info = {
                        'width': tex.width,
                        'height': tex.height,
                        'depth': tex.depth,
                        'mips': tex.mips,
                        'format': str(tex.format.Name()) if hasattr(tex.format, 'Name') else str(tex.format),
                        'type': str(tex.type) if hasattr(tex, 'type') else 'unknown'
                    }
                    self._texture_cache[rid] = info
                    return info
        except Exception:
            pass

        return None

    def _save_texture(self, resource_id, name: str) -> Optional[Path]:
        """Save texture data to file."""
        try:
            filepath = self.output_dir / f"{name}.png"

            texsave = rd.TextureSave()
            texsave.resourceId = resource_id
            texsave.mip = 0
            texsave.slice.sliceIndex = 0
            texsave.alpha = rd.AlphaMapping.Preserve
            texsave.destType = rd.FileType.PNG

            self.controller.SaveTexture(texsave, str(filepath))

            if filepath.exists():
                return filepath

        except Exception as e:
            print(f"  Warning: Failed to save texture {name}: {e}")

        return None

    def _extract_parameters(self, state, reflection, stage) -> List[MaterialParameter]:
        """Extract material parameters from constant buffers."""
        parameters = []

        if not reflection or not reflection.constantBlocks:
            return parameters

        pipe = state.GetGraphicsPipelineObject()

        for cb_idx, cb in enumerate(reflection.constantBlocks):
            try:
                # Get constant buffer contents
                cb_vars = self.controller.GetCBufferVariableContents(
                    pipe,
                    state.GetShader(stage),
                    stage,
                    reflection.entryPoint,
                    cb_idx,
                    rd.ResourceId.Null(),  # Use bound CB
                    0, 0
                )

                # Parse variables
                for var in cb_vars:
                    param = self._parse_cbuffer_variable(var)
                    if param:
                        parameters.append(param)

            except Exception:
                continue

        return parameters

    def _parse_cbuffer_variable(self, var) -> Optional[MaterialParameter]:
        """Parse a constant buffer variable into MaterialParameter."""
        try:
            name = var.name

            # Skip common non-material variables
            skip_patterns = ['view', 'proj', 'world', 'time', 'screen', 'camera']
            if any(p in name.lower() for p in skip_patterns):
                return None

            # Get value based on type
            if var.type.members:
                # Struct - skip or recurse
                return None

            # Get float values
            value = None
            type_name = "unknown"

            if var.type.columns == 1:
                if var.type.rows == 1:
                    value = var.value.f.x
                    type_name = "float"
                elif var.type.rows == 2:
                    value = [var.value.f.x, var.value.f.y]
                    type_name = "float2"
                elif var.type.rows == 3:
                    value = [var.value.f.x, var.value.f.y, var.value.f.z]
                    type_name = "float3"
                elif var.type.rows == 4:
                    value = [var.value.f.x, var.value.f.y, var.value.f.z, var.value.f.w]
                    type_name = "float4"

            if value is not None:
                return MaterialParameter(name=name, value=value, type=type_name)

        except Exception:
            pass

        return None

    def _detect_blend_mode(self, state) -> str:
        """Detect blend mode from pipeline state."""
        try:
            # Get blend state
            om = state.GetOutputMerger()

            if hasattr(om, 'blendState') and om.blendState:
                blend = om.blendState
                if hasattr(blend, 'blends') and blend.blends:
                    b = blend.blends[0]
                    if hasattr(b, 'enabled') and b.enabled:
                        return "blend"

            # Check depth write
            if hasattr(om, 'depthState'):
                if not om.depthState.depthEnable:
                    return "transparent"

        except Exception:
            pass

        return "opaque"


def extract_all_materials(controller, draw_events: List[int],
                          output_dir: str = "export/Materials") -> Dict[int, ExtractedMaterial]:
    """
    Extract materials from multiple draw calls.

    Args:
        controller: RenderDoc ReplayController
        draw_events: List of draw call event IDs
        output_dir: Output directory for textures

    Returns:
        Dict mapping event_id -> ExtractedMaterial
    """
    extractor = MaterialExtractor(controller, output_dir)
    materials = {}

    for event_id in draw_events:
        try:
            material = extractor.extract_draw_call(event_id)
            if material:
                materials[event_id] = material
        except Exception as e:
            print(f"  Warning: Failed to extract material at event {event_id}: {e}")

    return materials


def deduplicate_materials(materials: Dict[int, ExtractedMaterial]) -> List[ExtractedMaterial]:
    """
    Deduplicate materials based on texture bindings.

    Materials with the same texture set are considered duplicates.
    """
    unique = {}

    for event_id, mat in materials.items():
        # Create signature from texture resource IDs
        tex_sig = tuple(sorted(t.resource_id for t in mat.textures))

        if tex_sig not in unique:
            unique[tex_sig] = mat
        # else: duplicate, skip

    return list(unique.values())


# Common UE4 material parameter names
UE4_MATERIAL_PARAMS = {
    'BaseColor': TextureSlot.ALBEDO,
    'Normal': TextureSlot.NORMAL,
    'Roughness': TextureSlot.ROUGHNESS,
    'Metallic': TextureSlot.METALLIC,
    'AmbientOcclusion': TextureSlot.AO,
    'Emissive': TextureSlot.EMISSIVE,
    'Opacity': TextureSlot.OPACITY,
    'ORM': TextureSlot.ORM,
}


def create_pbr_material_from_extracted(
    extracted: ExtractedMaterial,
    texture_dir: str
) -> dict:
    """
    Convert ExtractedMaterial to PBR material definition.

    Returns dict suitable for glTF or USD material creation.
    """
    # Find textures by slot type
    texture_map = {t.texture_type: t for t in extracted.textures}

    pbr = {
        'name': extracted.name,
        'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
        'metallicFactor': 0.0,
        'roughnessFactor': 1.0,
        'doubleSided': extracted.is_double_sided,
        'alphaMode': 'BLEND' if extracted.is_transparent else 'OPAQUE',
        'textures': {}
    }

    # Map textures
    if TextureSlot.ALBEDO in texture_map:
        pbr['textures']['baseColorTexture'] = texture_map[TextureSlot.ALBEDO].filepath

    if TextureSlot.NORMAL in texture_map:
        pbr['textures']['normalTexture'] = texture_map[TextureSlot.NORMAL].filepath

    if TextureSlot.METALLIC in texture_map:
        pbr['textures']['metallicTexture'] = texture_map[TextureSlot.METALLIC].filepath
        pbr['metallicFactor'] = 1.0

    if TextureSlot.ROUGHNESS in texture_map:
        pbr['textures']['roughnessTexture'] = texture_map[TextureSlot.ROUGHNESS].filepath
        pbr['roughnessFactor'] = 1.0

    if TextureSlot.ORM in texture_map:
        # Packed ORM texture
        pbr['textures']['occlusionTexture'] = texture_map[TextureSlot.ORM].filepath
        pbr['textures']['metallicRoughnessTexture'] = texture_map[TextureSlot.ORM].filepath

    if TextureSlot.AO in texture_map:
        pbr['textures']['occlusionTexture'] = texture_map[TextureSlot.AO].filepath

    if TextureSlot.EMISSIVE in texture_map:
        pbr['textures']['emissiveTexture'] = texture_map[TextureSlot.EMISSIVE].filepath
        pbr['emissiveFactor'] = [1.0, 1.0, 1.0]

    # Parse scalar parameters
    for param in extracted.parameters:
        name_lower = param.name.lower()
        if 'metallic' in name_lower and isinstance(param.value, (int, float)):
            pbr['metallicFactor'] = float(param.value)
        elif 'roughness' in name_lower and isinstance(param.value, (int, float)):
            pbr['roughnessFactor'] = float(param.value)
        elif 'basecolor' in name_lower and isinstance(param.value, list):
            pbr['baseColorFactor'] = param.value + [1.0] * (4 - len(param.value))

    return pbr

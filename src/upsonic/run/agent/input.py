from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.messages.messages import (
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        ModelRequest,
    )


@dataclass
class AgentRunInput:
    """Input data for an agent run.
    
    Handles user prompts, images, and documents. URL types (ImageUrl, DocumentUrl)
    are automatically converted to BinaryContent for unified handling.
    """
    
    user_prompt: Union[str, List, Dict, "ModelRequest", "BaseModel", List["ModelRequest"]]
    images: Optional[Union[List["BinaryContent"], List[str]]] = None
    documents: Optional[Union[List["BinaryContent"], List[str]]] = None
    input: Optional[Union[str, List[Union[str, "BinaryContent"]]]] = None
    
    def __post_init__(self):
        """Convert any URL types to BinaryContent after initialization."""
        if self.images:
            self.images = [
                self._convert_url_to_binary(img) if self._is_url_type(img) else img
                for img in self.images
            ]
        if self.documents:
            self.documents = [
                self._convert_url_to_binary(doc) if self._is_url_type(doc) else doc
                for doc in self.documents
            ]
    
    @staticmethod
    def _is_url_type(content: Any) -> bool:
        """Check if content is a URL type that needs conversion."""
        from upsonic.messages.messages import DocumentUrl, ImageUrl
        return isinstance(content, (ImageUrl, DocumentUrl))
    
    @staticmethod
    def _convert_url_to_binary(url_content: Union["ImageUrl", "DocumentUrl"]) -> "BinaryContent":
        """Download URL content and convert to BinaryContent."""
        import httpx
        from upsonic.messages.messages import BinaryContent
        
        response = httpx.get(url_content.url)
        response.raise_for_status()
        return BinaryContent(
            data=response.content,
            media_type=url_content.media_type,
            identifier=url_content.identifier
        )
    
    @staticmethod
    async def _aconvert_url_to_binary(url_content: Union["ImageUrl", "DocumentUrl"]) -> "BinaryContent":
        """Async download URL content and convert to BinaryContent."""
        import httpx
        from upsonic.messages.messages import BinaryContent
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url_content.url)
            response.raise_for_status()
            return BinaryContent(
                data=response.content,
                media_type=url_content.media_type,
                identifier=url_content.identifier
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with raw values (no serialization)."""
        return {
            "user_prompt": self.user_prompt,
            "images": self.images,
            "documents": self.documents,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunInput":
        """Reconstruct from dictionary."""
        from upsonic.messages.messages import BinaryContent
        
        # Handle images (dict only)
        images = None
        images_data = data.get("images")
        if images_data:
            images = []
            for img in images_data:
                if isinstance(img, dict):
                    images.append(BinaryContent(
                        data=base64.b64decode(img["data"]) if isinstance(img["data"], str) else img["data"],
                        media_type=img.get("media_type", "application/octet-stream"),
                        identifier=img.get("identifier")
                    ))
        
        # Handle documents (dict only)
        documents = None
        documents_data = data.get("documents")
        if documents_data:
            documents = []
            for doc in documents_data:
                if isinstance(doc, dict):
                    documents.append(BinaryContent(
                        data=base64.b64decode(doc["data"]) if isinstance(doc["data"], str) else doc["data"],
                        media_type=doc.get("media_type", "application/octet-stream"),
                        identifier=doc.get("identifier")
                    ))
        
        return cls(
            user_prompt=data.get("user_prompt"),
            images=images,
            documents=documents
        )
    
    def build_input(self, context_formatted: Optional[str] = None) -> None:
        """
        Build the final input list from user_prompt, images, and documents.
        
        Processes file paths in images and documents into BinaryContent,
        combines with user_prompt, and sets self.input attribute.
        
        Args:
            context_formatted: Optional formatted context string to append to user_prompt
        """
        from upsonic.messages.messages import BinaryContent
        
        final_description = self.user_prompt
        if context_formatted and isinstance(context_formatted, str):
            if isinstance(final_description, str):
                final_description += "\n" + context_formatted
            else:
                final_description = str(final_description) + "\n" + context_formatted
        
        processed_images: List[BinaryContent] = []
        processed_documents: List[BinaryContent] = []
        
        if self.images:
            for img in self.images:
                if isinstance(img, str):
                    try:
                        with open(img, 'rb') as f:
                            data = f.read()
                        
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(img)
                        if mime_type is None:
                            mime_type = "application/octet-stream"
                        
                        binary_content = BinaryContent(
                            data=data,
                            media_type=mime_type,
                            identifier=img
                        )
                        processed_images.append(binary_content)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Failed to load image {img}: {e}", "AgentRunInput")
                elif isinstance(img, BinaryContent):
                    processed_images.append(img)
        
        if self.documents:
            for doc in self.documents:
                if isinstance(doc, str):
                    try:
                        with open(doc, 'rb') as f:
                            data = f.read()
                        
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(doc)
                        if mime_type is None:
                            mime_type = "application/octet-stream"
                        
                        binary_content = BinaryContent(
                            data=data,
                            media_type=mime_type,
                            identifier=doc
                        )
                        processed_documents.append(binary_content)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Failed to load document {doc}: {e}", "AgentRunInput")
                elif isinstance(doc, BinaryContent):
                    processed_documents.append(doc)
        
        if not processed_images and not processed_documents:
            self.input = final_description
        else:
            input_list: List[Union[str, BinaryContent]] = [final_description]
            input_list.extend(processed_images)
            input_list.extend(processed_documents)
            self.input = input_list
    
    async def abuild_input(self, context_formatted: Optional[str] = None) -> None:
        """
        Async version of build_input.
        
        Args:
            context_formatted: Optional formatted context string to append to user_prompt
        """
        from upsonic.messages.messages import BinaryContent
        
        final_description = self.user_prompt
        if context_formatted and isinstance(context_formatted, str):
            if isinstance(final_description, str):
                final_description += "\n" + context_formatted
            else:
                final_description = str(final_description) + "\n" + context_formatted
        
        processed_images: List[BinaryContent] = []
        processed_documents: List[BinaryContent] = []
        
        if self.images:
            for img in self.images:
                if isinstance(img, str):
                    try:
                        import aiofiles
                        async with aiofiles.open(img, 'rb') as f:
                            data = await f.read()
                        
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(img)
                        if mime_type is None:
                            mime_type = "application/octet-stream"
                        
                        binary_content = BinaryContent(
                            data=data,
                            media_type=mime_type,
                            identifier=img
                        )
                        processed_images.append(binary_content)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Failed to load image {img}: {e}", "AgentRunInput")
                elif isinstance(img, BinaryContent):
                    processed_images.append(img)
        
        if self.documents:
            for doc in self.documents:
                if isinstance(doc, str):
                    try:
                        import aiofiles
                        async with aiofiles.open(doc, 'rb') as f:
                            data = await f.read()
                        
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(doc)
                        if mime_type is None:
                            mime_type = "application/octet-stream"
                        
                        binary_content = BinaryContent(
                            data=data,
                            media_type=mime_type,
                            identifier=doc
                        )
                        processed_documents.append(binary_content)
                    except Exception as e:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Failed to load document {doc}: {e}", "AgentRunInput")
                elif isinstance(doc, BinaryContent):
                    processed_documents.append(doc)
        
        if not processed_images and not processed_documents:
            self.input = final_description
        else:
            input_list: List[Union[str, BinaryContent]] = [final_description]
            input_list.extend(processed_images)
            input_list.extend(processed_documents)
            self.input = input_list
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        import cloudpickle
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentRunInput":
        """Deserialize from bytes."""
        import cloudpickle
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)


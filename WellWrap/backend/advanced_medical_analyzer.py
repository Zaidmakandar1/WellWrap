"""
Advanced Medical Analyzer - Lightweight Version
Integration layer between Flask backend and medical analysis
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)

class MedicalTestResult:
    """Medical test result data structure"""
    def __init__(self, test_name, value, unit=None, normal_range=None, status=None, severity_score=0.0):
        self.test_name = test_name
        self.value = value
        self.unit = unit
        self.normal_range = normal_range
        self.status = status
        self.severity_score = severity_score

@dataclass
class DiseaseRisk:
    """Disease risk assessment result"""
    disease_name: str
    risk_level: str
    confidence: float
    contributing_factors: List[str]
    description: str
    recommendations: List[str]

class AdvancedMedicalAnalyzer:
    """
    Lightweight Medical Analyzer for Flask Backend
    Provides basic medical report analysis without heavy ML dependencies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized lightweight medical analyzer")
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from uploaded files (PDF or image) with robust fallbacks."""
        try:
            # Read file content
            from io import BytesIO
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            if not file_content:
                return ""
            
            # Quick type detection
            is_pdf = file_content[:5] == b"%PDF-"
            is_png = file_content[:8] == b"\x89PNG\r\n\x1a\n"
            is_jpg = file_content[:3] == b"\xff\xd8\xff"
            
            # Route images to image OCR handler
            if is_png or is_jpg:
                return self.extract_text_from_image(file)
            
            # PDF extraction flow with layered fallbacks
            self.logger.info("Starting comprehensive PDF text extraction...")
            
            # Method 1: Try PyPDF2 first (fastest for text-based PDFs)
            text = self._extract_with_pypdf2(file_content)
            if text and len(text.strip()) >= 30:
                return self._clean_extracted_text(text.strip())
            
            # Method 2: Try pdfplumber (better for tables and complex layouts)
            text = self._extract_with_pdfplumber(file_content)
            if text and len(text.strip()) >= 30:
                return self._clean_extracted_text(text.strip())
            
            # Method 3: Try pymupdf (fitz) - excellent for various PDF types
            text = self._extract_with_pymupdf(file_content)
            if text and len(text.strip()) >= 30:
                return self._clean_extracted_text(text.strip())
            
            # Method 4: Try OCR for scanned PDFs (comprehensive approach)
            text = self._extract_with_comprehensive_ocr(file_content, 'pdf')
            if text and len(text.strip()) >= 15:  # Lower threshold for OCR
                return self._clean_extracted_text(text.strip())
            
            # Method 5: Try alternative PDF libraries
            text = self._extract_with_alternative_methods(file_content)
            if text and len(text.strip()) >= 15:
                return self._clean_extracted_text(text.strip())
            
            # Method 6: Last resort - try to extract any text at all
            all_methods_text = self._extract_any_text_possible(file_content)
            if all_methods_text and len(all_methods_text.strip()) >= 10:
                return self._clean_extracted_text(all_methods_text)
            
            return "Unable to extract readable text from this PDF. The document may be encrypted, corrupted, or contain only images without text."
            
        except Exception as e:
            self.logger.error(f"File text extraction failed: {e}")
            return f"Error extracting text: {str(e)}"
    
    def extract_text_from_image(self, file) -> str:
        """Extract text from uploaded image file using fast OCR methods"""
        try:
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            self.logger.info("Starting fast image text extraction...")
            
            # Try simple OCR first (fastest)
            text = self._extract_with_simple_ocr(file)
            if text and len(text.strip()) >= 15:
                self.logger.info(f"Simple OCR successful: {len(text)} characters")
                return self._clean_extracted_text(text.strip())
            
            # Try EasyOCR (doesn't require system Tesseract)
            file.seek(0)
            easyocr_text = self._extract_with_easyocr_simple(file)
            if easyocr_text and len(easyocr_text.strip()) >= 15:
                self.logger.info(f"EasyOCR successful: {len(easyocr_text)} characters")
                return self._clean_extracted_text(easyocr_text.strip())
            
            # Try basic Tesseract with preprocessing
            file.seek(0)
            tesseract_text = self._extract_with_tesseract_basic(file)
            if tesseract_text and len(tesseract_text.strip()) >= 15:
                self.logger.info(f"Basic Tesseract successful: {len(tesseract_text)} characters")
                return self._clean_extracted_text(tesseract_text.strip())
            
            # Use the best result we got, even if short
            best_text = max([text or "", easyocr_text or "", tesseract_text or ""], key=len)
            if best_text and len(best_text.strip()) > 5:
                self.logger.info(f"Using best available result: {len(best_text)} characters")
                return self._clean_extracted_text(best_text.strip())
            
            return self._clean_extracted_text(text.strip()) if text else "No readable text found in image. Please ensure the image contains clear, readable text."
            
        except Exception as e:
            self.logger.error(f"Image text extraction failed: {e}")
            return f"Error extracting text from image: {str(e)}"
    
    def _extract_with_pymupdf(self, file_content: bytes) -> str:
        """Extract text using PyMuPDF (fitz) - excellent for various PDF types"""
        try:
            import fitz  # PyMuPDF
            from io import BytesIO
            
            # Open PDF document
            doc = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text.strip())
                
                # Also try to extract text blocks for better formatting
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) >= 5 and block[4].strip():  # block[4] contains text
                        block_text = block[4].strip()
                        if block_text not in page_text:  # Avoid duplicates
                            text_parts.append(block_text)
            
            doc.close()
            return "\n\n".join(text_parts).strip()
            
        except ImportError:
            self.logger.warning("PyMuPDF (fitz) not available")
            return ""
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _extract_with_alternative_methods(self, file_content: bytes) -> str:
        """Try alternative PDF extraction methods"""
        text = ""
        
        # Method 1: Try PDFMiner
        try:
            from pdfminer.high_level import extract_text
            from io import BytesIO
            
            text = extract_text(BytesIO(file_content))
            if text and len(text.strip()) > 20:
                return text.strip()
        except ImportError:
            self.logger.warning("PDFMiner not available")
        except Exception as e:
            self.logger.warning(f"PDFMiner extraction failed: {e}")
        
        # Method 2: Try Tika (if available)
        try:
            from tika import parser
            from io import BytesIO
            
            parsed = parser.from_buffer(file_content)
            if parsed and 'content' in parsed and parsed['content']:
                text = parsed['content'].strip()
                if len(text) > 20:
                    return text
        except ImportError:
            self.logger.warning("Tika not available")
        except Exception as e:
            self.logger.warning(f"Tika extraction failed: {e}")
        
        return text
    
    def _extract_with_tesseract_enhanced(self, file_content: bytes, file_type: str) -> str:
        """Enhanced Tesseract OCR with multiple configurations and preprocessing"""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO
            
            if file_type == 'pdf':
                # Convert PDF to images with multiple DPI settings
                images = []
                try:
                    import pdf2image
                    # Try different DPI settings for better results
                    for dpi in [300, 200, 150]:
                        try:
                            imgs = pdf2image.convert_from_bytes(
                                file_content, 
                                dpi=dpi,
                                first_page=1,
                                last_page=min(10, 5)  # Limit pages for performance
                            )
                            images.extend(imgs)
                            break  # Use first successful DPI
                        except Exception as e:
                            self.logger.warning(f"PDF conversion failed at {dpi} DPI: {e}")
                            continue
                except ImportError:
                    self.logger.warning("pdf2image not available for enhanced Tesseract")
                    return ""
            else:
                # Direct image processing
                images = [Image.open(BytesIO(file_content))]
            
            if not images:
                return ""
            
            best_text = ""
            
            for i, image in enumerate(images):
                try:
                    # Try multiple preprocessing approaches
                    preprocessing_methods = [
                        lambda img: img,  # Original image
                        lambda img: self._preprocess_image_for_ocr(img),  # Standard preprocessing
                        lambda img: self._preprocess_image_aggressive(img),  # Aggressive preprocessing
                        lambda img: self._preprocess_image_medical(img),  # Medical document specific
                    ]
                    
                    page_best_text = ""
                    
                    for preprocess_func in preprocessing_methods:
                        try:
                            processed_image = preprocess_func(image)
                            
                            # Try different Tesseract configurations
                            configs = [
                                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,():/\\-+= %',
                                '--psm 4 --oem 3',
                                '--psm 3 --oem 1',
                                '--psm 1 --oem 3',
                                '--psm 6 --oem 1',
                                '--psm 11 --oem 3',  # Sparse text
                                '--psm 13 --oem 3',  # Raw line
                            ]
                            
                            for config in configs:
                                try:
                                    text = pytesseract.image_to_string(processed_image, config=config)
                                    if text and len(text.strip()) > len(page_best_text.strip()):
                                        page_best_text = text.strip()
                                except Exception as e:
                                    continue
                                    
                        except Exception as e:
                            continue
                    
                    if page_best_text and len(page_best_text) > 10:
                        best_text += f"\nPage {i+1}:\n{page_best_text}\n"
                        
                except Exception as e:
                    self.logger.warning(f"Enhanced Tesseract failed on page {i+1}: {e}")
                    continue
            
            return best_text.strip()
            
        except ImportError:
            self.logger.warning("Enhanced Tesseract OCR not available")
            return ""
        except Exception as e:
            self.logger.warning(f"Enhanced Tesseract OCR extraction failed: {e}")
            return ""
    
    def _extract_with_paddleocr(self, file_content: bytes, file_type: str) -> str:
        """Extract text using PaddleOCR (alternative OCR engine)"""
        try:
            from paddleocr import PaddleOCR
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Initialize PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
            if file_type == 'pdf':
                # Convert PDF to images
                try:
                    import pdf2image
                    images = pdf2image.convert_from_bytes(
                        file_content, 
                        dpi=200,
                        first_page=1,
                        last_page=5
                    )
                except ImportError:
                    return ""
            else:
                images = [Image.open(BytesIO(file_content))]
            
            text = ""
            for i, image in enumerate(images):
                try:
                    # Convert PIL image to numpy array
                    image_array = np.array(image)
                    
                    # Extract text using PaddleOCR
                    results = ocr.ocr(image_array, cls=True)
                    
                    # Combine detected text
                    page_text = ""
                    if results and results[0]:
                        for line in results[0]:
                            if line and len(line) >= 2:
                                detected_text = line[1][0]  # Text content
                                confidence = line[1][1]     # Confidence score
                                if confidence > 0.6:  # Only high-confidence results
                                    page_text += detected_text + " "
                    
                    if page_text.strip():
                        text += f"Page {i+1}:\n{page_text.strip()}\n\n"
                        
                except Exception as e:
                    self.logger.warning(f"PaddleOCR failed on page {i+1}: {e}")
                    continue
            
            return text.strip()
            
        except ImportError:
            self.logger.warning("PaddleOCR not available")
            return ""
        except Exception as e:
            self.logger.warning(f"PaddleOCR extraction failed: {e}")
            return ""
    
    def _extract_with_trocr(self, file_content: bytes, file_type: str) -> str:
        """Extract text using TrOCR (Transformer-based OCR)"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            from io import BytesIO
            
            # Load TrOCR model (printed text version)
            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            
            if file_type == 'pdf':
                # Convert PDF to images
                try:
                    import pdf2image
                    images = pdf2image.convert_from_bytes(
                        file_content, 
                        dpi=150,  # Lower DPI for TrOCR (it's more efficient)
                        first_page=1,
                        last_page=3  # Limit for performance
                    )
                except ImportError:
                    return ""
            else:
                images = [Image.open(BytesIO(file_content))]
            
            text = ""
            for i, image in enumerate(images):
                try:
                    # Preprocess image for TrOCR
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize if too large (TrOCR works better with smaller images)
                    width, height = image.size
                    if width > 1024 or height > 1024:
                        ratio = min(1024/width, 1024/height)
                        new_size = (int(width * ratio), int(height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Process with TrOCR
                    pixel_values = processor(images=image, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if generated_text.strip():
                        text += f"Page {i+1}:\n{generated_text.strip()}\n\n"
                        
                except Exception as e:
                    self.logger.warning(f"TrOCR failed on page {i+1}: {e}")
                    continue
            
            return text.strip()
            
        except ImportError:
            self.logger.warning("TrOCR not available (requires transformers library)")
            return ""
        except Exception as e:
            self.logger.warning(f"TrOCR extraction failed: {e}")
            return ""
    
    def _preprocess_image_aggressive(self, image):
        """Aggressive image preprocessing for difficult documents"""
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import numpy as np
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Significantly increase size for better OCR
            width, height = image.size
            if width < 1500 or height < 1500:
                scale_factor = max(1500/width, 1500/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Aggressive contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Aggressive sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Multiple noise reduction passes
            image = image.filter(ImageFilter.MedianFilter(size=3))
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Aggressive threshold
            try:
                import cv2
                img_array = np.array(image)
                
                # Multiple threshold approaches
                binary1 = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 15, 3
                )
                
                binary2 = cv2.threshold(
                    img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                
                # Combine both approaches
                combined = cv2.bitwise_and(binary1, binary2)
                
                # Morphological operations to clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                
                image = Image.fromarray(combined)
            except ImportError:
                # Fallback without OpenCV
                threshold = 100  # More aggressive threshold
                image = image.point(lambda p: p > threshold and 255)
            
            return image
        except Exception as e:
            self.logger.warning(f"Aggressive preprocessing failed: {e}")
            return image
    
    def _preprocess_image_medical(self, image):
        """Medical document specific preprocessing"""
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import numpy as np
            
            # Convert to RGB first
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Medical documents often need larger size
            width, height = image.size
            if width < 1200 or height < 1200:
                scale_factor = max(1200/width, 1200/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Medical documents often have specific contrast issues
            image = ImageOps.autocontrast(image, cutoff=2)
            
            # Moderate contrast enhancement (medical docs can be sensitive)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Light sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Gentle noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Medical document optimized threshold
            try:
                import cv2
                img_array = np.array(image)
                
                # Use adaptive threshold optimized for medical documents
                binary = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 13, 4
                )
                
                # Light morphological cleaning
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                image = Image.fromarray(binary)
            except ImportError:
                # Fallback threshold for medical documents
                threshold = 120
                image = image.point(lambda p: p > threshold and 255)
            
            return image
        except Exception as e:
            self.logger.warning(f"Medical preprocessing failed: {e}")
            return image
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s.,():/\-+=% ]', '', text)
        
        # Fix common OCR mistakes in medical context
        replacements = {
            'O': '0',  # Common in numbers
            'l': '1',  # In numeric contexts
            'S': '5',  # Sometimes in numbers
        }
        
        # Apply replacements carefully (only in numeric contexts)
        text = re.sub(r'\b(\w*[A-Za-z]\w*)\s*:\s*([O|l|S]+\.?\d*)', 
                     lambda m: m.group(1) + ': ' + ''.join(replacements.get(c, c) for c in m.group(2)), 
                     text)
        
        return text.strip()
    
    def _extract_any_text_possible(self, file_content: bytes) -> str:
        """Last resort method to extract any possible text"""
        text_parts = []
        
        # Method 1: Try PyPDF2 with different approaches
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            for page in pdf_reader.pages:
                try:
                    # Standard extraction
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                    
                    # Try extracting from page objects
                    if hasattr(page, '_objects'):
                        for obj in page._objects.values():
                            if hasattr(obj, 'get_data'):
                                try:
                                    data = obj.get_data()
                                    if isinstance(data, bytes):
                                        decoded = data.decode('utf-8', errors='ignore')
                                        if any(char.isalnum() for char in decoded) and len(decoded) > 10:
                                            text_parts.append(decoded)
                                except:
                                    continue
                except:
                    continue
        except:
            pass
        
        # Method 2: Try to extract as raw text
        try:
            # Sometimes PDFs have readable text in raw bytes
            raw_text = file_content.decode('utf-8', errors='ignore')
            # Look for patterns that might be text
            import re
            potential_text = re.findall(r'[A-Za-z][A-Za-z\s]{10,}', raw_text)
            if potential_text:
                text_parts.extend(potential_text)
        except:
            pass
        
        return "\n".join(text_parts).strip()
    
    def _extract_with_alternative_image_methods(self, file_content: bytes) -> str:
        """Try alternative image processing methods for text extraction"""
        try:
            from PIL import Image
            from io import BytesIO
            
            image = Image.open(BytesIO(file_content))
            best_text = ""
            
            # Method 1: Try different image formats/modes
            try:
                # Convert to different modes and try OCR
                modes = ['L', 'RGB', '1']  # Grayscale, RGB, Binary
                for mode in modes:
                    try:
                        converted_image = image.convert(mode)
                        text = self._simple_tesseract_ocr(converted_image)
                        if text and len(text.strip()) > len(best_text.strip()):
                            best_text = text
                    except:
                        continue
            except:
                pass
            
            # Method 2: Try image enhancement techniques
            try:
                enhanced_images = self._create_enhanced_variants(image)
                for enhanced_img in enhanced_images:
                    try:
                        text = self._simple_tesseract_ocr(enhanced_img)
                        if text and len(text.strip()) > len(best_text.strip()):
                            best_text = text
                    except:
                        continue
            except:
                pass
            
            return best_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Alternative image methods failed: {e}")
            return ""
    
    def _simple_tesseract_ocr(self, image) -> str:
        """Simple Tesseract OCR without complex preprocessing"""
        try:
            import pytesseract
            return pytesseract.image_to_string(image, config='--psm 6').strip()
        except:
            return ""
    
    def _create_enhanced_variants(self, image):
        """Create multiple enhanced variants of an image for OCR"""
        variants = []
        
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            
            # Original image
            variants.append(image)
            
            # High contrast version
            try:
                enhancer = ImageEnhance.Contrast(image)
                variants.append(enhancer.enhance(2.0))
            except:
                pass
            
            # High brightness version
            try:
                enhancer = ImageEnhance.Brightness(image)
                variants.append(enhancer.enhance(1.5))
            except:
                pass
            
            # Sharpened version
            try:
                variants.append(image.filter(ImageFilter.SHARPEN))
            except:
                pass
            
            # Edge enhanced version
            try:
                variants.append(image.filter(ImageFilter.EDGE_ENHANCE))
            except:
                pass
            
            # Inverted version (sometimes helps with dark backgrounds)
            try:
                if image.mode == 'L':
                    variants.append(ImageOps.invert(image))
                elif image.mode == 'RGB':
                    variants.append(ImageOps.invert(image.convert('L')))
            except:
                pass
            
        except Exception as e:
            self.logger.warning(f"Failed to create enhanced variants: {e}")
        
        return variants
    
    def _extract_with_simple_ocr(self, file) -> str:
        """Simple and fast OCR extraction for images"""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Load image
            image = Image.open(BytesIO(file_content))
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Simple preprocessing
            if image.mode != 'L':
                image = image.convert('L')  # Convert to grayscale
            
            # Simple OCR with basic config
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            self.logger.info(f"Simple OCR extracted {len(text)} characters")
            return text.strip()
            
        except ImportError:
            self.logger.warning("Tesseract not available for simple OCR")
            return ""
        except Exception as e:
            self.logger.warning(f"Simple OCR failed: {e}")
            return ""
    
    def _extract_with_tesseract_basic(self, file) -> str:
        """Basic Tesseract OCR with minimal preprocessing"""
        try:
            import pytesseract
            from PIL import Image, ImageEnhance
            from io import BytesIO
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Load and preprocess image
            image = Image.open(BytesIO(file_content))
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Basic enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Try basic OCR
            text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
            
            self.logger.info(f"Basic Tesseract extracted {len(text)} characters")
            return text.strip()
            
        except ImportError:
            self.logger.warning("Tesseract not available for basic OCR")
            return ""
        except Exception as e:
            self.logger.warning(f"Basic Tesseract failed: {e}")
            return ""
    
    def _extract_with_easyocr_simple(self, file) -> str:
        """Simple EasyOCR extraction (doesn't require system Tesseract)"""
        try:
            import easyocr
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], verbose=False)
            
            # Load image
            image = Image.open(BytesIO(file_content))
            
            # Convert PIL image to numpy array for EasyOCR
            image_array = np.array(image)
            
            # Extract text using EasyOCR
            results = reader.readtext(image_array)
            
            # Combine all detected text
            text_parts = []
            for (bbox, detected_text, confidence) in results:
                if confidence > 0.5:  # Only include high-confidence results
                    text_parts.append(detected_text)
            
            text = " ".join(text_parts)
            self.logger.info(f"Simple EasyOCR extracted {len(text)} characters")
            return text.strip()
            
        except ImportError:
            self.logger.warning("EasyOCR not available")
            return ""
        except Exception as e:
            self.logger.warning(f"Simple EasyOCR failed: {e}")
            return ""
    
    def _extract_with_enhanced_preprocessing(self, file_content: bytes) -> str:
        """Extract text with enhanced image preprocessing"""
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            from io import BytesIO
            import cv2
            import numpy as np
            
            # Load image
            image = Image.open(BytesIO(file_content))
            
            # Convert to OpenCV format for advanced preprocessing
            img_array = np.array(image.convert('RGB'))
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply multiple preprocessing techniques
            preprocessing_methods = [
                # Method 1: Gaussian blur + threshold
                lambda img: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                
                # Method 2: Morphological operations
                lambda img: cv2.morphologyEx(
                    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                ),
                
                # Method 3: Adaptive threshold
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                
                # Method 4: Bilateral filter + threshold
                lambda img: cv2.threshold(cv2.bilateralFilter(img, 9, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ]
            
            best_text = ""
            for i, method in enumerate(preprocessing_methods):
                try:
                    processed_img = method(img_gray)
                    pil_img = Image.fromarray(processed_img)
                    
                    # Extract text
                    text = pytesseract.image_to_string(pil_img, config='--psm 6')
                    
                    # Keep the best result (longest meaningful text)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        
                except Exception as e:
                    self.logger.warning(f"Preprocessing method {i+1} failed: {e}")
                    continue
            
            return best_text.strip()
            
        except ImportError:
            self.logger.warning("OpenCV not available for enhanced preprocessing")
            return ""
        except Exception as e:
            self.logger.warning(f"Enhanced preprocessing failed: {e}")
            return ""
    
    def _extract_with_pypdf2(self, file_content: bytes) -> str:
        """Extract text using PyPDF2"""
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
        except ImportError:
            self.logger.warning("PyPDF2 not available")
            return ""
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, file_content: bytes) -> str:
        """Extract text using pdfplumber (better for tables and complex layouts)"""
        try:
            import pdfplumber
            from io import BytesIO
            
            text = ""
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Also extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text += " | ".join([cell or "" for cell in row]) + "\n"
            
            return text.strip()
        except ImportError:
            self.logger.warning("pdfplumber not available")
            return ""
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_with_comprehensive_ocr(self, file_content: bytes, file_type: str) -> str:
        """Extract text using multiple OCR methods with enhanced preprocessing"""
        best_text = ""
        
        # Method 1: Try Tesseract OCR with multiple configurations
        tesseract_text = self._extract_with_tesseract_enhanced(file_content, file_type)
        if tesseract_text and len(tesseract_text.strip()) > len(best_text.strip()):
            best_text = tesseract_text
        
        # Method 2: Try EasyOCR
        easyocr_text = self._extract_with_easyocr(file_content, file_type)
        if easyocr_text and len(easyocr_text.strip()) > len(best_text.strip()):
            best_text = easyocr_text
        
        # Method 3: Try PaddleOCR if available
        paddle_text = self._extract_with_paddleocr(file_content, file_type)
        if paddle_text and len(paddle_text.strip()) > len(best_text.strip()):
            best_text = paddle_text
        
        # Method 4: Try TrOCR (Transformer-based OCR) if available
        trocr_text = self._extract_with_trocr(file_content, file_type)
        if trocr_text and len(trocr_text.strip()) > len(best_text.strip()):
            best_text = trocr_text
        
        return best_text.strip()
    
    def _extract_with_tesseract(self, file_content: bytes, file_type: str) -> str:
        """Extract text using Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO
            
            if file_type == 'pdf':
                # Convert PDF to images first
                try:
                    import pdf2image
                    # Use higher DPI for better OCR results
                    images = pdf2image.convert_from_bytes(
                        file_content, 
                        dpi=300,  # Higher DPI for better text recognition
                        first_page=1,
                        last_page=10  # Limit to first 10 pages for performance
                    )
                except ImportError:
                    self.logger.warning("pdf2image not available for PDF OCR")
                    return ""
            else:
                # Direct image processing
                images = [Image.open(BytesIO(file_content))]
            
            text = ""
            for i, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    processed_image = self._preprocess_image_for_ocr(image)
                    
                    # Try different PSM modes for better results
                    psm_modes = ['--psm 6', '--psm 4', '--psm 3', '--psm 1']
                    page_text = ""
                    
                    for psm in psm_modes:
                        try:
                            page_text = pytesseract.image_to_string(
                                processed_image, 
                                config=f'{psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,():/\\-+= '
                            )
                            if page_text and len(page_text.strip()) > 20:
                                break
                        except:
                            continue
                    
                    if page_text:
                        text += f"Page {i+1}:\n{page_text}\n\n"
                        
                except Exception as e:
                    self.logger.warning(f"Tesseract failed on page {i+1}: {e}")
                    continue
            
            return text.strip()
        except ImportError:
            self.logger.warning("Tesseract OCR not available")
            return ""
        except Exception as e:
            self.logger.warning(f"Tesseract OCR extraction failed: {e}")
            return ""
    
    def _extract_with_easyocr(self, file_content: bytes, file_type: str) -> str:
        """Extract text using EasyOCR (alternative OCR engine)"""
        try:
            import easyocr
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])
            
            if file_type == 'pdf':
                # Convert PDF to images first
                try:
                    import pdf2image
                    images = pdf2image.convert_from_bytes(
                        file_content, 
                        dpi=200,  # Good balance of quality and performance
                        first_page=1,
                        last_page=5  # Limit for performance
                    )
                except ImportError:
                    self.logger.warning("pdf2image not available for EasyOCR")
                    return ""
            else:
                # Direct image processing
                images = [Image.open(BytesIO(file_content))]
            
            text = ""
            for i, image in enumerate(images):
                try:
                    # Convert PIL image to numpy array for EasyOCR
                    image_array = np.array(image)
                    
                    # Extract text using EasyOCR
                    results = reader.readtext(image_array)
                    
                    # Combine all detected text
                    page_text = ""
                    for (bbox, detected_text, confidence) in results:
                        if confidence > 0.5:  # Only include high-confidence results
                            page_text += detected_text + " "
                    
                    if page_text:
                        text += f"Page {i+1}:\n{page_text}\n\n"
                        
                except Exception as e:
                    self.logger.warning(f"EasyOCR failed on page {i+1}: {e}")
                    continue
            
            return text.strip()
        except ImportError:
            self.logger.warning("EasyOCR not available")
            return ""
        except Exception as e:
            self.logger.warning(f"EasyOCR extraction failed: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image):
        """Advanced image preprocessing to improve OCR accuracy"""
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import numpy as np
            
            # Convert to RGB first if needed
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Resize image if too small (OCR works better on larger images)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000/width, 1000/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Auto-contrast to improve text visibility
            image = ImageOps.autocontrast(image)
            
            # Enhance contrast further
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Optional: Apply threshold to create binary image for better OCR
            try:
                import cv2
                # Convert PIL to OpenCV format
                img_array = np.array(image)
                
                # Apply adaptive threshold
                binary = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Convert back to PIL
                image = Image.fromarray(binary)
            except ImportError:
                # Fallback: simple threshold using PIL
                threshold = 128
                image = image.point(lambda p: p > threshold and 255)
            
            return image
        except Exception as e:
            self.logger.warning(f"Advanced image preprocessing failed: {e}")
            # Return original image if preprocessing fails
            return image
    
    def extract_medical_data(self, text: str) -> List[MedicalTestResult]:
        """Extract medical test results from text using comprehensive pattern matching"""
        results = []
        
        # Enhanced test patterns with multiple variations
        patterns = {
            # Blood Count Tests
            'Hemoglobin': [
                r'(?:hemoglobin|hgb|hb)[\s:]*(\d+\.?\d*)',
                r'hb[\s:]*(\d+\.?\d*)',
                r'hemoglobin[\s:]*(\d+\.?\d*)\s*g/dl',
            ],
            'Hematocrit': [
                r'(?:hematocrit|hct|packed cell volume|pcv)[\s:]*(\d+\.?\d*)',
                r'hct[\s:]*(\d+\.?\d*)\s*%',
            ],
            'White Blood Cells': [
                r'(?:wbc|white blood cell|leucocyte)[\s:]*(\d+\.?\d*)',
                r'total.*wbc[\s:]*(\d+\.?\d*)',
                r'white.*cell.*count[\s:]*(\d+\.?\d*)',
            ],
            'Red Blood Cells': [
                r'(?:rbc|red blood cell|erythrocyte)[\s:]*(\d+\.?\d*)',
                r'red.*cell.*count[\s:]*(\d+\.?\d*)',
            ],
            'Platelets': [
                r'(?:platelets|plt|thrombocyte)[\s:]*(\d+\.?\d*)',
                r'platelet.*count[\s:]*(\d+\.?\d*)',
            ],
            
            # Lipid Panel
            'Total Cholesterol': [
                r'(?:total cholesterol|cholesterol total|cholesterol)[\s:]*(\d+\.?\d*)',
                r'chol[\s:]*(\d+\.?\d*)',
            ],
            'LDL Cholesterol': [
                r'(?:ldl cholesterol|ldl|low density lipoprotein)[\s:]*(\d+\.?\d*)',
                r'ldl-c[\s:]*(\d+\.?\d*)',
            ],
            'HDL Cholesterol': [
                r'(?:hdl cholesterol|hdl|high density lipoprotein)[\s:]*(\d+\.?\d*)',
                r'hdl-c[\s:]*(\d+\.?\d*)',
            ],
            'Triglycerides': [
                r'(?:triglycerides|trig|tg)[\s:]*(\d+\.?\d*)',
                r'triglyceride[\s:]*(\d+\.?\d*)',
            ],
            
            # Metabolic Panel
            'Glucose': [
                r'(?:glucose|blood sugar|fasting glucose|random glucose)[\s:]*(\d+\.?\d*)',
                r'fbs[\s:]*(\d+\.?\d*)',
                r'rbs[\s:]*(\d+\.?\d*)',
            ],
            'Creatinine': [
                r'(?:creatinine|creat|serum creatinine)[\s:]*(\d+\.?\d*)',
                r'cr[\s:]*(\d+\.?\d*)',
            ],
            'Blood Urea Nitrogen': [
                r'(?:bun|blood urea nitrogen|urea nitrogen)[\s:]*(\d+\.?\d*)',
                r'urea[\s:]*(\d+\.?\d*)',
            ],
            'Sodium': [
                r'(?:sodium|na\+|na)[\s:]*(\d+\.?\d*)',
                r'serum sodium[\s:]*(\d+\.?\d*)',
            ],
            'Potassium': [
                r'(?:potassium|k\+|k)[\s:]*(\d+\.?\d*)',
                r'serum potassium[\s:]*(\d+\.?\d*)',
            ],
            
            # Liver Function
            'ALT': [
                r'(?:alt|alanine aminotransferase|sgpt)[\s:]*(\d+\.?\d*)',
                r'alanine.*transaminase[\s:]*(\d+\.?\d*)',
            ],
            'AST': [
                r'(?:ast|aspartate aminotransferase|sgot)[\s:]*(\d+\.?\d*)',
                r'aspartate.*transaminase[\s:]*(\d+\.?\d*)',
            ],
            'Bilirubin Total': [
                r'(?:total bilirubin|bilirubin total|bilirubin)[\s:]*(\d+\.?\d*)',
                r't\.bil[\s:]*(\d+\.?\d*)',
            ],
            
            # Thyroid Function
            'TSH': [
                r'(?:tsh|thyroid stimulating hormone)[\s:]*(\d+\.?\d*)',
                r'thyrotropin[\s:]*(\d+\.?\d*)',
            ],
            'T3': [
                r'(?:t3|triiodothyronine)[\s:]*(\d+\.?\d*)',
                r'free t3[\s:]*(\d+\.?\d*)',
            ],
            'T4': [
                r'(?:t4|thyroxine)[\s:]*(\d+\.?\d*)',
                r'free t4[\s:]*(\d+\.?\d*)',
            ],
            
            # Cardiac Markers
            'Troponin': [
                r'(?:troponin|trop|cardiac troponin)[\s:]*(\d+\.?\d*)',
                r'troponin.*i[\s:]*(\d+\.?\d*)',
            ],
            
            # Inflammatory Markers
            'ESR': [
                r'(?:esr|erythrocyte sedimentation rate|sed rate)[\s:]*(\d+\.?\d*)',
                r'sedimentation.*rate[\s:]*(\d+\.?\d*)',
            ],
            'CRP': [
                r'(?:crp|c-reactive protein|c reactive protein)[\s:]*(\d+\.?\d*)',
                r'c.*reactive[\s:]*(\d+\.?\d*)',
            ],
        }
        
        text_lower = text.lower()
        
        for test_name, pattern_list in patterns.items():
            found = False
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    try:
                        value = float(matches[0])
                        status = self._determine_status(test_name, value)
                        severity = self._calculate_severity(test_name, value, status)
                        
                        results.append(MedicalTestResult(
                            test_name=test_name,
                            value=value,
                            unit=self._get_unit(test_name),
                            normal_range=self._get_normal_range(test_name),
                            status=status,
                            severity_score=severity
                        ))
                        found = True
                        break  # Found a match, move to next test
                    except ValueError:
                        continue
            
            if found:
                continue
        
        return results
    
    def _determine_status(self, test_name: str, value: float) -> str:
        """Determine if test result is normal, high, or low"""
        ranges = {
            'Hemoglobin': (12.0, 16.0),
            'Hematocrit': (36.0, 46.0),
            'White Blood Cells': (4.0, 11.0),
            'Platelets': (150.0, 450.0),
            'Glucose': (70.0, 100.0),
            'Total Cholesterol': (0.0, 200.0),
            'LDL Cholesterol': (0.0, 100.0),
            'HDL Cholesterol': (40.0, 999.0),  # Higher is better
            'Triglycerides': (0.0, 150.0),
            'Creatinine': (0.6, 1.3),
        }
        
        if test_name in ranges:
            low, high = ranges[test_name]
            if value < low:
                return 'low'
            elif value > high:
                return 'high'
        
        return 'normal'
    
    def _calculate_severity(self, test_name: str, value: float, status: str) -> float:
        """Calculate severity score (0.0 to 1.0)"""
        if status == 'normal':
            return 0.0
        
        # Define critical thresholds
        critical_ranges = {
            'Hemoglobin': {'low': 8.0, 'high': 18.0},
            'Glucose': {'low': 50.0, 'high': 200.0},
            'Total Cholesterol': {'high': 300.0},
            'Triglycerides': {'high': 500.0},
        }
        
        if test_name in critical_ranges:
            thresholds = critical_ranges[test_name]
            if status == 'low' and 'low' in thresholds and value <= thresholds['low']:
                return 0.8
            elif status == 'high' and 'high' in thresholds and value >= thresholds['high']:
                return 0.8
        
        return 0.5  # Moderate severity for other abnormal results
    
    def _get_unit(self, test_name: str) -> str:
        """Get typical units for tests"""
        units = {
            'Hemoglobin': 'g/dL',
            'Hematocrit': '%',
            'White Blood Cells': 'K/uL',
            'Platelets': 'K/uL',
            'Glucose': 'mg/dL',
            'Total Cholesterol': 'mg/dL',
            'LDL Cholesterol': 'mg/dL',
            'HDL Cholesterol': 'mg/dL',
            'Triglycerides': 'mg/dL',
            'Creatinine': 'mg/dL',
        }
        return units.get(test_name, '')
    
    def _get_normal_range(self, test_name: str) -> str:
        """Get normal ranges for tests"""
        ranges = {
            'Hemoglobin': '12.0-16.0 g/dL',
            'Hematocrit': '36-46%',
            'White Blood Cells': '4.0-11.0 K/uL',
            'Platelets': '150-450 K/uL',
            'Glucose': '70-100 mg/dL',
            'Total Cholesterol': '<200 mg/dL',
            'LDL Cholesterol': '<100 mg/dL',
            'HDL Cholesterol': '>40 mg/dL',
            'Triglycerides': '<150 mg/dL',
            'Creatinine': '0.6-1.3 mg/dL',
        }
        return ranges.get(test_name, 'Varies')
    
    def detect_diseases(self, test_results: List[MedicalTestResult]) -> List[DiseaseRisk]:
        """Detect potential diseases based on test results"""
        diseases = []
        
        # Create a dictionary for easier lookup
        test_dict = {result.test_name.lower(): result for result in test_results}
        
        # Anemia detection
        if 'hemoglobin' in test_dict and test_dict['hemoglobin'].status == 'low':
            severity = "high" if test_dict['hemoglobin'].value < 8.0 else "medium"
            diseases.append(DiseaseRisk(
                disease_name="Iron Deficiency Anemia",
                risk_level=severity,
                confidence=0.8,
                contributing_factors=["Low hemoglobin levels"],
                description="A condition where your blood has fewer red blood cells or less hemoglobin than normal",
                recommendations=[
                    "Consult with your doctor immediately",
                    "Consider iron-rich diet (spinach, red meat, beans)",
                    "Get iron studies and further testing",
                    "Monitor for symptoms like fatigue and weakness"
                ]
            ))
        
        # Cardiovascular risk detection
        high_cholesterol = 'total cholesterol' in test_dict and test_dict['total cholesterol'].status == 'high'
        high_ldl = 'ldl cholesterol' in test_dict and test_dict['ldl cholesterol'].status == 'high'
        low_hdl = 'hdl cholesterol' in test_dict and test_dict['hdl cholesterol'].status == 'low'
        high_triglycerides = 'triglycerides' in test_dict and test_dict['triglycerides'].status == 'high'
        
        if high_cholesterol or high_ldl or low_hdl or high_triglycerides:
            risk_factors = []
            if high_cholesterol: risk_factors.append("High total cholesterol")
            if high_ldl: risk_factors.append("High LDL cholesterol")
            if low_hdl: risk_factors.append("Low HDL cholesterol")
            if high_triglycerides: risk_factors.append("High triglycerides")
            
            risk_level = "high" if len(risk_factors) >= 2 else "medium"
            
            diseases.append(DiseaseRisk(
                disease_name="Cardiovascular Disease Risk",
                risk_level=risk_level,
                confidence=0.9,
                contributing_factors=risk_factors,
                description="Elevated lipid levels increase risk of heart disease and stroke",
                recommendations=[
                    "Follow heart-healthy diet (Mediterranean or DASH diet)",
                    "Increase physical activity (150 min/week moderate exercise)",
                    "Consider statin therapy (consult doctor)",
                    "Schedule cardiology consultation",
                    "Monitor blood pressure regularly"
                ]
            ))
        
        # Diabetes risk detection
        if 'glucose' in test_dict and test_dict['glucose'].status == 'high':
            glucose_value = test_dict['glucose'].value
            if glucose_value >= 126:
                risk_level = "high"
                disease_name = "Diabetes"
            elif glucose_value >= 100:
                risk_level = "medium"
                disease_name = "Prediabetes"
            else:
                risk_level = "low"
                disease_name = "Glucose Intolerance"
            
            diseases.append(DiseaseRisk(
                disease_name=disease_name,
                risk_level=risk_level,
                confidence=0.85,
                contributing_factors=["Elevated fasting glucose"],
                description="High blood sugar levels that may indicate diabetes or prediabetes",
                recommendations=[
                    "Monitor blood sugar regularly",
                    "Follow diabetic-friendly diet",
                    "Maintain healthy weight",
                    "Regular exercise",
                    "Consult endocrinologist"
                ]
            ))
        
        # Kidney function concerns
        if 'creatinine' in test_dict and test_dict['creatinine'].status == 'high':
            diseases.append(DiseaseRisk(
                disease_name="Kidney Function Impairment",
                risk_level="medium",
                confidence=0.7,
                contributing_factors=["Elevated creatinine"],
                description="Kidney function may be impaired based on elevated waste product levels",
                recommendations=[
                    "Stay well hydrated",
                    "Monitor kidney function regularly",
                    "Limit protein intake if advised",
                    "Consult nephrologist if levels remain high"
                ]
            ))
        
        return diseases
    
    def _generate_action_plan(self, test_results: List[MedicalTestResult], 
                              disease_risks: List[DiseaseRisk]) -> List[Dict[str, Any]]:
        """Translate findings into concrete actions grouped by priority."""
        actions: List[Dict[str, Any]] = []

        def add_action(priority: str, title: str, details: str, when: str, who: str, links: Optional[List[Dict[str, str]]] = None):
            actions.append({
                'priority': priority,  # immediate, soon, routine
                'title': title,
                'details': details,
                'when': when,
                'who': who,
                'links': links or []
            })

        # Map abnormal tests to actions
        for tr in test_results:
            if tr.status == 'high':
                if tr.test_name in ['Total Cholesterol', 'LDL Cholesterol', 'Triglycerides']:
                    add_action(
                        priority='soon',
                        title='Improve lipid profile',
                        details='Adopt heart-healthy diet, increase activity; discuss statin therapy if applicable.',
                        when='Within 2-4 weeks',
                        who='Primary care / cardiology'
                    )
                if tr.test_name == 'Glucose':
                    add_action(
                        priority='soon',
                        title='Address elevated blood sugar',
                        details='Reduce refined carbs, monitor fasting glucose; evaluate for diabetes if persistent.',
                        when='Within 1-2 weeks',
                        who='Primary care / endocrinology'
                    )
                if tr.test_name == 'Creatinine':
                    add_action(
                        priority='soon',
                        title='Assess kidney function',
                        details='Repeat kidney panel, review hydration, medications; consider nephrology referral.',
                        when='Within 2-4 weeks',
                        who='Primary care / nephrology'
                    )
            elif tr.status == 'low':
                if tr.test_name == 'Hemoglobin':
                    add_action(
                        priority='soon',
                        title='Evaluate anemia',
                        details='Order iron studies; consider iron-rich diet and supplementation per clinician.',
                        when='Within 1-2 weeks',
                        who='Primary care / hematology'
                    )

        # Map disease risks to actions
        for risk in disease_risks:
            if risk.disease_name == 'Cardiovascular Disease Risk':
                add_action(
                    priority='soon' if risk.risk_level in ['high', 'medium'] else 'routine',
                    title='Reduce cardiovascular risk',
                    details='Manage lipids, BP, weight; stop smoking; consider statin; 150 min/week exercise.',
                    when='Start this week',
                    who='Primary care / cardiology'
                )
            if risk.disease_name in ['Diabetes', 'Prediabetes']:
                add_action(
                    priority='immediate' if risk.disease_name == 'Diabetes' else 'soon',
                    title='Glycemic management plan',
                    details='Dietary changes, glucose monitoring; consider medication and educator referral.',
                    when='Within 1 week' if risk.disease_name == 'Diabetes' else 'Within 2-4 weeks',
                    who='Primary care / endocrinology'
                )
            if 'Kidney' in risk.disease_name:
                add_action(
                    priority='soon',
                    title='Kidney protection steps',
                    details='Optimize hydration, review nephrotoxic meds; consider eGFR/urine ACR testing.',
                    when='Within 2-4 weeks',
                    who='Primary care / nephrology'
                )

        # De-duplicate by title keeping highest priority
        priority_rank = {'immediate': 0, 'soon': 1, 'routine': 2}
        dedup = {}
        for a in actions:
            t = a['title']
            if t not in dedup or priority_rank[a['priority']] < priority_rank[dedup[t]['priority']]:
                dedup[t] = a
        actions = sorted(dedup.values(), key=lambda x: priority_rank.get(x['priority'], 3))
        return actions

    def generate_health_summary(self, test_results: List[MedicalTestResult], 
                               disease_risks: List[DiseaseRisk]) -> Dict[str, Any]:
        """Generate comprehensive health summary with consequences if ignored"""
        total_tests = len(test_results)
        normal_tests = sum(1 for t in test_results if t.status == 'normal')
        abnormal_tests = total_tests - normal_tests
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct points for abnormal results
        if total_tests > 0:
            abnormal_ratio = abnormal_tests / total_tests
            health_score -= abnormal_ratio * 30
        
        # Deduct points for disease risks
        for risk in disease_risks:
            if risk.risk_level == 'high':
                health_score -= 25
            elif risk.risk_level == 'medium':
                health_score -= 15
            else:
                health_score -= 5
        
        health_score = max(0, min(100, int(health_score)))
        
        # Determine overall status with detailed descriptions
        if health_score >= 85:
            overall_status = "Excellent"
            status_color = "success"
            status_description = "Your test results look great! Keep up the good work with your health."
        elif health_score >= 70:
            overall_status = "Good"
            status_color = "success"
            status_description = "Most of your test results are within normal ranges with minor areas for improvement."
        elif health_score >= 55:
            overall_status = "Fair"
            status_color = "warning"
            status_description = "Some test results need attention. Consider lifestyle changes and follow-up with your doctor."
        else:
            overall_status = "Needs Attention"
            status_color = "danger"
            status_description = "Several test results are concerning. Please consult with your healthcare provider promptly."
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(test_results, disease_risks)
        
        # Generate user-friendly explanations
        test_explanations = self._generate_test_explanations(test_results)
        
        high_risk_items = [d for d in disease_risks if d.risk_level == 'high']
        medium_risk_items = [d for d in disease_risks if d.risk_level == 'medium']
        
        # Build narrative summary
        key_findings = []
        for tr in test_results:
            if tr.status != 'normal':
                key_findings.append(f"{tr.test_name}: {tr.value} {tr.unit or ''} (expected {tr.normal_range}, {tr.status})")
        for dr in disease_risks:
            key_findings.append(f"Risk: {dr.disease_name} ({dr.risk_level})")
        
        consequences = []
        for dr in high_risk_items:
            if 'cardiovascular' in dr.disease_name.lower():
                consequences.append("Increased likelihood of heart attack or stroke if untreated")
            if 'diabetes' in dr.disease_name.lower():
                consequences.append("Risk of nerve damage, kidney disease, and vision loss if uncontrolled")
            if 'anemia' in dr.disease_name.lower():
                consequences.append("Severe fatigue, shortness of breath, and potential cardiac strain if untreated")
            if 'kidney' in dr.disease_name.lower():
                consequences.append("Progression to chronic kidney disease if not addressed")
        if medium_risk_items and not consequences:
            consequences.append("Condition may worsen over time and increase long-term complications")
        if not consequences and abnormal_tests > 0:
            consequences.append("Abnormal results may progress without follow-up care")

        action_plan = self._generate_action_plan(test_results, disease_risks)
        
        narrative = {
            'summary_text': (
                f"We analyzed {total_tests} tests. Your health score is {health_score}/100 "
                f"({overall_status}). "
                + ("Key findings include: " + "; ".join(key_findings) + ". " if key_findings else "")
                + ("If recommended actions are not taken, potential consequences include: "
                   + "; ".join(consequences) + "." if consequences else "")
            ).strip()
        }
        
        return {
            'health_score': health_score,
            'overall_status': overall_status,
            'status_color': status_color,
            'status_description': status_description,
            'total_tests': total_tests,
            'normal_tests': normal_tests,
            'abnormal_tests': abnormal_tests,
            'disease_risks_count': len(disease_risks),
            'high_risk_diseases': [d.disease_name for d in disease_risks if d.risk_level == 'high'],
            'medium_risk_diseases': [d.disease_name for d in disease_risks if d.risk_level == 'medium'],
            'recommendations': self._generate_general_recommendations(test_results, disease_risks),
            'summary': f"WellWrap analyzed {total_tests} test results from your medical report. Your overall health score is {health_score}/100 ({overall_status}). {status_description}",
            'next_steps': self._generate_next_steps(disease_risks),
            'detailed_analysis': detailed_analysis,
            'test_explanations': test_explanations,
            'key_findings': self._generate_key_findings(test_results, disease_risks),
            'lifestyle_tips': self._generate_lifestyle_tips(test_results, disease_risks),
            'narrative': narrative,
            'consequences_if_ignored': consequences,
            'action_plan': action_plan
        }
    
    def _generate_detailed_analysis(self, test_results: List[MedicalTestResult], 
                                   disease_risks: List[DiseaseRisk]) -> Dict[str, Any]:
        """Generate detailed analysis by body system"""
        analysis = {
            'blood_health': {'tests': [], 'status': 'normal', 'summary': ''},
            'heart_health': {'tests': [], 'status': 'normal', 'summary': ''},
            'metabolic_health': {'tests': [], 'status': 'normal', 'summary': ''},
            'liver_health': {'tests': [], 'status': 'normal', 'summary': ''},
            'kidney_health': {'tests': [], 'status': 'normal', 'summary': ''},
            'thyroid_health': {'tests': [], 'status': 'normal', 'summary': ''},
        }
        
        # Categorize tests by body system
        blood_tests = ['Hemoglobin', 'Hematocrit', 'White Blood Cells', 'Red Blood Cells', 'Platelets', 'ESR']
        heart_tests = ['Total Cholesterol', 'LDL Cholesterol', 'HDL Cholesterol', 'Triglycerides', 'Troponin']
        metabolic_tests = ['Glucose', 'Sodium', 'Potassium']
        liver_tests = ['ALT', 'AST', 'Bilirubin Total']
        kidney_tests = ['Creatinine', 'Blood Urea Nitrogen']
        thyroid_tests = ['TSH', 'T3', 'T4']
        
        for test in test_results:
            # Convert MedicalTestResult to dictionary for JSON serialization
            test_dict = {
                'test_name': test.test_name,
                'value': test.value,
                'unit': test.unit,
                'normal_range': test.normal_range,
                'status': test.status,
                'severity_score': test.severity_score
            }
            
            if test.test_name in blood_tests:
                analysis['blood_health']['tests'].append(test_dict)
            elif test.test_name in heart_tests:
                analysis['heart_health']['tests'].append(test_dict)
            elif test.test_name in metabolic_tests:
                analysis['metabolic_health']['tests'].append(test_dict)
            elif test.test_name in liver_tests:
                analysis['liver_health']['tests'].append(test_dict)
            elif test.test_name in kidney_tests:
                analysis['kidney_health']['tests'].append(test_dict)
            elif test.test_name in thyroid_tests:
                analysis['thyroid_health']['tests'].append(test_dict)
        
        # Generate summaries for each system
        for system, data in analysis.items():
            if data['tests']:
                abnormal_count = sum(1 for t in data['tests'] if t['status'] != 'normal')
                total_count = len(data['tests'])
                
                if abnormal_count == 0:
                    data['status'] = 'normal'
                    data['summary'] = f"All {total_count} {system.replace('_', ' ')} tests are within normal ranges."
                elif abnormal_count <= total_count / 2:
                    data['status'] = 'mild_concern'
                    data['summary'] = f"{abnormal_count} of {total_count} {system.replace('_', ' ')} tests need attention."
                else:
                    data['status'] = 'concern'
                    data['summary'] = f"Multiple {system.replace('_', ' ')} tests are abnormal and require medical attention."
        
        return analysis
    
    def _generate_test_explanations(self, test_results: List[MedicalTestResult]) -> Dict[str, str]:
        """Generate user-friendly explanations for each test"""
        explanations = {
            'Hemoglobin': "Hemoglobin carries oxygen in your blood. Low levels may indicate anemia, while high levels could suggest dehydration or other conditions.",
            'Hematocrit': "This measures the percentage of red blood cells in your blood. It helps diagnose anemia and other blood disorders.",
            'White Blood Cells': "These cells fight infections. High levels may indicate infection or inflammation, while low levels could mean weakened immunity.",
            'Platelets': "Platelets help your blood clot. Low levels increase bleeding risk, while high levels may increase clotting risk.",
            'Total Cholesterol': "This measures all cholesterol in your blood. High levels increase heart disease risk.",
            'LDL Cholesterol': "Often called 'bad' cholesterol, high LDL levels can clog arteries and increase heart attack risk.",
            'HDL Cholesterol': "Known as 'good' cholesterol, higher HDL levels protect against heart disease.",
            'Triglycerides': "High levels of these blood fats increase heart disease risk and may indicate metabolic problems.",
            'Glucose': "Blood sugar levels. High levels may indicate diabetes or prediabetes.",
            'Creatinine': "This waste product indicates kidney function. High levels suggest kidney problems.",
            'ALT': "A liver enzyme. High levels may indicate liver damage or disease.",
            'AST': "Another liver enzyme. Elevated levels can indicate liver or heart problems.",
            'TSH': "Thyroid stimulating hormone controls your thyroid. Abnormal levels indicate thyroid problems.",
        }
        
        result_explanations = {}
        for test in test_results:
            if test.test_name in explanations:
                base_explanation = explanations[test.test_name]
                
                if test.status == 'high':
                    result_explanations[test.test_name] = f"{base_explanation} Your level ({test.value} {test.unit}) is higher than normal ({test.normal_range})."
                elif test.status == 'low':
                    result_explanations[test.test_name] = f"{base_explanation} Your level ({test.value} {test.unit}) is lower than normal ({test.normal_range})."
                else:
                    result_explanations[test.test_name] = f"{base_explanation} Your level ({test.value} {test.unit}) is within the normal range ({test.normal_range})."
        
        return result_explanations
    
    def _generate_key_findings(self, test_results: List[MedicalTestResult], 
                              disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate key findings in simple language"""
        findings = []
        
        # Critical findings
        critical_tests = [t for t in test_results if t.severity_score >= 0.8]
        if critical_tests:
            findings.append(f"🚨 Critical: {len(critical_tests)} test result(s) require immediate medical attention")
        
        # High-risk diseases
        high_risks = [r for r in disease_risks if r.risk_level == 'high']
        if high_risks:
            findings.append(f"⚠️ High Risk: Potential for {', '.join([r.disease_name for r in high_risks])}")
        
        # Positive findings
        normal_tests = [t for t in test_results if t.status == 'normal']
        if len(normal_tests) >= len(test_results) * 0.7:
            findings.append(f"✅ Good News: {len(normal_tests)} out of {len(test_results)} tests are normal")
        
        # Specific findings
        if any(t.test_name == 'Hemoglobin' and t.status == 'low' for t in test_results):
            findings.append("🩸 Low hemoglobin detected - possible anemia")
        
        if any(t.test_name in ['Total Cholesterol', 'LDL Cholesterol'] and t.status == 'high' for t in test_results):
            findings.append("💓 High cholesterol detected - heart health needs attention")
        
        if any(t.test_name == 'Glucose' and t.status == 'high' for t in test_results):
            findings.append("🍯 High blood sugar detected - diabetes risk")
        
        return findings
    
    def _generate_lifestyle_tips(self, test_results: List[MedicalTestResult], 
                                disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate personalized lifestyle tips"""
        tips = []
        
        # General tips
        tips.append("🥗 Eat a balanced diet rich in fruits, vegetables, and whole grains")
        tips.append("🏃‍♂️ Aim for at least 150 minutes of moderate exercise per week")
        tips.append("💧 Stay hydrated by drinking plenty of water")
        tips.append("😴 Get 7-9 hours of quality sleep each night")
        
        # Specific tips based on results
        if any(t.test_name == 'Hemoglobin' and t.status == 'low' for t in test_results):
            tips.append("🥩 Include iron-rich foods like lean meat, spinach, and beans")
            tips.append("🍊 Eat vitamin C-rich foods to improve iron absorption")
        
        if any(t.test_name in ['Total Cholesterol', 'LDL Cholesterol'] and t.status == 'high' for t in test_results):
            tips.append("🐟 Choose heart-healthy fats like those in fish, nuts, and olive oil")
            tips.append("🚫 Limit saturated and trans fats")
            tips.append("🌾 Increase fiber intake with oats, beans, and whole grains")
        
        if any(t.test_name == 'Glucose' and t.status == 'high' for t in test_results):
            tips.append("🍎 Choose complex carbohydrates over simple sugars")
            tips.append("⚖️ Maintain a healthy weight")
            tips.append("🚶‍♀️ Take short walks after meals to help control blood sugar")
        
        if any(t.test_name == 'Creatinine' and t.status == 'high' for t in test_results):
            tips.append("💧 Stay well-hydrated to support kidney function")
            tips.append("🧂 Limit sodium intake")
        
        return tips[:8]  # Limit to 8 tips
    
    def convert_to_json_serializable(self, test_results: List[MedicalTestResult], 
                                   disease_risks: List[DiseaseRisk]) -> Dict[str, Any]:
        """Convert MedicalTestResult and DiseaseRisk objects to JSON-serializable format"""
        
        # Convert test results to dictionaries
        test_results_dict = []
        for test in test_results:
            test_results_dict.append({
                'test_name': test.test_name,
                'value': test.value,
                'unit': test.unit,
                'normal_range': test.normal_range,
                'status': test.status,
                'severity_score': test.severity_score
            })
        
        # Convert disease risks to dictionaries
        disease_risks_dict = []
        for risk in disease_risks:
            disease_risks_dict.append({
                'disease_name': risk.disease_name,
                'risk_level': risk.risk_level,
                'confidence': risk.confidence,
                'contributing_factors': risk.contributing_factors,
                'description': risk.description,
                'recommendations': risk.recommendations
            })
        
        return {
            'test_results': test_results_dict,
            'disease_risks': disease_risks_dict
        }
    
    def _generate_general_recommendations(self, test_results: List[MedicalTestResult], 
                                        disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate general health recommendations"""
        recommendations = set()  # Use set to avoid duplicates
        
        # Add recommendations from disease risks
        for risk in disease_risks:
            recommendations.update(risk.recommendations[:2])  # Add first 2 recommendations
        
        # Add general health recommendations
        recommendations.update([
            "Maintain a balanced diet rich in fruits and vegetables",
            "Stay physically active with regular exercise",
            "Get adequate sleep (7-9 hours per night)",
            "Schedule regular check-ups with your healthcare provider"
        ])
        
        return list(recommendations)[:8]  # Limit to 8 recommendations
    
    def _generate_next_steps(self, disease_risks: List[DiseaseRisk]) -> List[str]:
        """Generate immediate next steps based on risk assessment"""
        next_steps = []
        
        high_risks = [r for r in disease_risks if r.risk_level == 'high']
        medium_risks = [r for r in disease_risks if r.risk_level == 'medium']
        
        if high_risks:
            next_steps.append("🚨 Schedule immediate appointment with your doctor")
            next_steps.append("📋 Discuss high-risk findings and treatment options")
        
        if medium_risks:
            next_steps.append("📅 Schedule follow-up appointment within 2-4 weeks")
            next_steps.append("📊 Consider additional testing as recommended")
        
        if not high_risks and not medium_risks:
            next_steps.append("✅ Continue current health maintenance routine")
            next_steps.append("📅 Schedule routine check-up in 6-12 months")
        
        next_steps.append("📱 Monitor symptoms and contact doctor if concerns arise")
        
        return next_steps
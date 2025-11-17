from altair import Description
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableStructureOptions)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
import os, tempfile

OUTPUT_DIR = "/home/tupham/Documents/Development/Student_Handbook/ouput"

def load_data_from_upload(file_object) -> tuple:
    """
    Load dữ liệu từ uploaded file (Streamlit UploadedFile) và convert sang Markdown
    
    Returns:
        (markdown_text, doc_name): Tuple chứa markdown text và tên document
    """
    
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,   
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    doc_name = getattr(file_object, 'name', 'unknown_document')
    if doc_name.endswith('.pdf'):
        doc_name = doc_name[:-4]
    
    # File temp
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_object.read())
            tmp_path = tmp_file.name
        
        # Convert từ file path
        result = converter.convert(tmp_path)
        
        # Xóa file tạm
        os.unlink(tmp_path)
        
    except Exception as e:
        raise Exception(f"Lỗi khi convert PDF: {e}")
    
    
    doc = result.document
    if not doc:
        raise Exception("Docling không thể xử lý file này!")
    
    # Export sang Markdown
    markdown_text = doc.export_to_markdown()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, f"{doc_name}.md")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Đã lưu Markdown: {output_file_path}")

    return markdown_text, doc_name
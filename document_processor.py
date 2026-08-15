import os
import pickle
import faiss
import numpy as np
import torch
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from config import load_config

import pymupdf
import easyocr
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

INDEX_PATH = "faiss_index.idx"
CHUNK_MAPPING_PATH = "index_to_chunk.pkl"
METADATA_PATH = "document_metadata.pkl"
UPLOAD_BASE_DIR = "uploads"

# Global caches
_model_cache = {}
_tokenizer_cache = {}
_ocr_reader = None
SEARCH_CACHE = {}

def clear_search_cache():
    global SEARCH_CACHE
    SEARCH_CACHE.clear()

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def get_model_and_tokenizer(model_name=None):
    if model_name is None:
        config = load_config()
        model_name = config.get("model_repo_id", "distilbert-base-uncased")
    
    if model_name not in _model_cache:
        try:
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
            _model_cache[model_name] = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            model_name = "distilbert-base-uncased"
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
            _model_cache[model_name] = AutoModel.from_pretrained(model_name)
    
    return _tokenizer_cache[model_name], _model_cache[model_name]

def get_embedding(text, pooling='mean'):
    tokenizer, model = get_model_and_tokenizer()
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(input_ids)
    if pooling == 'mean':
        return output.last_hidden_state.mean(dim=1).numpy()
    elif pooling == 'max':
        return output.last_hidden_state.max(dim=1)[0].numpy()
    return output.last_hidden_state.mean(dim=1).numpy()

def extract_text_from_image(file_path):
    try:
        reader = get_ocr_reader()
        result = reader.readtext(file_path, detail=0)
        return [{'page_number': 1, 'text': '\n'.join(result)}]
    except Exception as e:
        print(f"Error OCR image: {e}")
        return []

def extract_text_from_pdf(file_path):
    pages_text = []
    try:
        doc = pymupdf.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                # OCR fallback
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                reader = get_ocr_reader()
                result = reader.readtext(img_data, detail=0)
                text = '\n'.join(result)
            if text:
                pages_text.append({'page_number': page_num, 'text': text})
    except Exception as e:
        print(f"Error reading PDF with OCR fallback: {e}")
        # fallback to PyPDF2
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append({'page_number': page_num, 'text': page_text})
        except Exception as e2:
            print(f"Failed fallback: {e2}")
    return pages_text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = DocxDocument(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return [{'page_number': 1, 'text': text}]

def extract_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return [{'page_number': 1, 'text': text}]

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return extract_text_from_image(file_path)
    return []

def chunk_text(text, chunk_size=500, overlap=50):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if current_length + len(sentence_words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = []
            overlap_len = 0
            for s in reversed(current_chunk):
                s_words = s.split()
                if overlap_len + len(s_words) <= overlap:
                    overlap_words.insert(0, s)
                    overlap_len += len(s_words)
                else:
                    break
            current_chunk = overlap_words
            current_length = overlap_len
            
        current_chunk.append(sentence)
        current_length += len(sentence_words)
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def initialize_or_load_index():
    config = load_config()
    dimension = config.get("dimension", 768)
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_MAPPING_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNK_MAPPING_PATH, "rb") as f:
            index_to_chunk = pickle.load(f)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
        else:
            metadata = {"documents": [], "total_chunks": 0}
    else:
        index = faiss.IndexFlatL2(dimension)
        index_to_chunk = {}
        metadata = {"documents": [], "total_chunks": 0}
    return index, index_to_chunk, metadata

def add_document_to_index(file_path, filename, original_filename):
    clear_search_cache()
    config = load_config()
    chunk_size = config.get("chunk_size", 500)
    overlap = config.get("chunk_overlap", 50)
    
    pages_text = extract_text_from_file(file_path)
    if not pages_text:
        return False, "Could not extract text from document", None
    
    index, index_to_chunk, metadata = initialize_or_load_index()
    start_idx = len(index_to_chunk)
    embeddings = []
    timestamp = datetime.now()
    doc_id = f"doc_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(metadata['documents'])}"
    chunk_counter = 0
    
    for page_data in pages_text:
        page_num = page_data['page_number']
        page_text = page_data['text']
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        
        for chunk in page_chunks:
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            index_to_chunk[start_idx + chunk_counter] = {
                "text": chunk,
                "document": original_filename,
                "doc_id": doc_id,
                "chunk_index": chunk_counter,
                "page_number": page_num
            }
            chunk_counter += 1
    
    if len(embeddings) == 0:
        return False, "No content to index", None
    
    embeddings_array = np.vstack(embeddings).astype('float32')
    index.add(embeddings_array)
    
    file_ext = os.path.splitext(original_filename)[1].lower()
    doc_type = "PDF" if file_ext == ".pdf" else "Word" if file_ext == ".docx" else "Image" if file_ext in ['.png', '.jpg'] else "Text"
    
    doc_metadata = {
        "id": doc_id,
        "filename": original_filename,
        "path": file_path,
        "chunks": chunk_counter,
        "uploaded_on": timestamp.isoformat(),
        "type": doc_type,
        "size": os.path.getsize(file_path),
        "pages": len(pages_text)
    }
    
    metadata["documents"].append(doc_metadata)
    metadata["total_chunks"] = len(index_to_chunk)
    
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNK_MAPPING_PATH, "wb") as f:
        pickle.dump(index_to_chunk, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    return True, f"Successfully indexed {chunk_counter} chunks from {original_filename}", doc_id

def delete_document(doc_id):
    clear_search_cache()
    config = load_config()
    dimension = config.get("dimension", 768)
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNK_MAPPING_PATH):
        return False, "No index found"
    
    index, index_to_chunk, metadata = initialize_or_load_index()
    doc_to_delete = None
    for doc in metadata["documents"]:
        if doc["id"] == doc_id:
            doc_to_delete = doc
            break
    if not doc_to_delete:
        return False, "Document not found"
    
    new_index = faiss.IndexFlatL2(dimension)
    new_index_to_chunk = {}
    new_idx = 0
    embeddings_to_keep = []
    
    for idx, chunk_data in index_to_chunk.items():
        if chunk_data.get("doc_id") != doc_id:
            new_index_to_chunk[new_idx] = chunk_data
            vector = get_embedding(chunk_data["text"])
            embeddings_to_keep.append(vector)
            new_idx += 1
    
    if embeddings_to_keep:
        embeddings_array = np.vstack(embeddings_to_keep).astype('float32')
        new_index.add(embeddings_array)
    
    metadata["documents"] = [doc for doc in metadata["documents"] if doc["id"] != doc_id]
    metadata["total_chunks"] = len(new_index_to_chunk)
    
    if os.path.exists(doc_to_delete["path"]):
        try:
            os.remove(doc_to_delete["path"])
        except:
            pass
    
    faiss.write_index(new_index, INDEX_PATH)
    with open(CHUNK_MAPPING_PATH, "wb") as f:
        pickle.dump(new_index_to_chunk, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    return True, f"Successfully deleted {doc_to_delete['filename']}"

def get_document_content(doc_id):
    metadata = get_metadata()
    for doc in metadata["documents"]:
        if doc["id"] == doc_id:
            if os.path.exists(doc["path"]):
                pages_text = extract_text_from_file(doc["path"])
                full_text = "\n\n".join([page['text'] for page in pages_text])
                return {
                    "filename": doc["filename"],
                    "content": full_text,
                    "type": doc["type"],
                    "pages": doc.get("pages", 1)
                }
    return None

def expand_query(query):
    expanded = set(query.lower().split())
    for word in query.split():
        try:
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    syn_word = l.name().replace('_', ' ')
                    if len(syn_word.split()) == 1:
                        expanded.add(syn_word.lower())
        except:
            pass
    # Limit expanded query size
    return " ".join(list(expanded)[:10])

def highlight_text(text, query):
    words = query.lower().split()
    for word in words:
        if len(word) > 2:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    return text

def search_in_index(query, num_matches=5, sort_by="relevance"):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNK_MAPPING_PATH):
        return []
    if not query or not query.strip():
        return []
    
    cache_key = f"{query}_{num_matches}_{sort_by}"
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNK_MAPPING_PATH, "rb") as f:
        index_to_chunk = pickle.load(f)
    
    if len(index_to_chunk) == 0:
        return []
        
    config = load_config()
    top_k = config.get("top_k", 10)
    
    # 1. FAISS Semantic Search
    vector = get_embedding(query)
    faiss_k = min(top_k * 3, len(index_to_chunk))
    D, I = index.search(vector.reshape(1, -1), faiss_k)
    
    # 2. Hybrid Keyword Search (BM25)
    expanded_q = expand_query(query)
    corpus = [index_to_chunk[idx]["text"] for idx in range(len(index_to_chunk))]
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(expanded_q.split(" "))
    
    # 3. Combine Scores
    scored_candidates = {}
    max_d = max(D[0]) if len(D[0]) > 0 else 1 # max dimension in vector comparison
        
    for d, idx in zip(D[0], I[0]):
        if idx < len(index_to_chunk):
            sim = 1.0 - (d / (max_d + 1e-10))
            scored_candidates[idx] = {'semantic': sim}
            
    max_bm25 = max(bm25_scores) if len(bm25_scores)>0 else 1
    if max_bm25 == 0: max_bm25 = 1
    
    for idx in range(len(bm25_scores)):
        if bm25_scores[idx] > 0 or idx in scored_candidates:
            if idx not in scored_candidates:
                scored_candidates[idx] = {'semantic': 0.0}
            scored_candidates[idx]['keyword'] = bm25_scores[idx] / max_bm25
            
    results_list = []
    for idx, scores in scored_candidates.items():
        hybrid_score = (scores['semantic'] * 0.5) + (scores.get('keyword', 0.0) * 0.5)
        if hybrid_score > 0:
            results_list.append((hybrid_score, idx))
            
    results_list.sort(key=lambda x: x[0], reverse=True)
    results_list = results_list[:top_k]
    
    metadata = get_metadata()
    final_results = []
    
    for score, idx in results_list:
        chunk_data = index_to_chunk[idx]
        upload_date = None
        for doc in metadata.get("documents", []):
            if doc["id"] == chunk_data.get("doc_id"):
                upload_date = doc.get("uploaded_on")
                break
        
        highlighted = highlight_text(chunk_data["text"], query)
        
        final_results.append({
            "text": highlighted,
            "document": chunk_data.get("document", "Unknown"),
            "doc_id": chunk_data.get("doc_id"),
            "page_number": chunk_data.get("page_number", 1),
            "chunk_number": chunk_data.get("chunk_index", 0) + 1,
            "score": float(1.0 - score), 
            "uploaded_on": upload_date
        })
        
    if sort_by == "recent":
        final_results.sort(key=lambda x: x.get("uploaded_on", ""), reverse=True)
        
    final_results = final_results[:num_matches]
    SEARCH_CACHE[cache_key] = final_results
    return final_results

def get_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {"documents": [], "total_chunks": 0}

def rebuild_index_with_new_config():
    clear_search_cache()
    metadata = get_metadata()
    config = load_config()
    dimension = config.get("dimension", 768)
    
    if len(metadata.get("documents", [])) == 0:
        return True, "No documents to reindex"
    
    new_index = faiss.IndexFlatL2(dimension)
    new_index_to_chunk = {}
    chunk_counter = 0
    embeddings = []
    
    for doc in metadata["documents"]:
        if not os.path.exists(doc["path"]):
            continue
        pages_text = extract_text_from_file(doc["path"])
        
        for page_data in pages_text:
            page_num = page_data['page_number']
            page_text = page_data['text']
            page_chunks = chunk_text(page_text, config.get("chunk_size", 500), config.get("chunk_overlap", 50))
            
            for idx, chunk in enumerate(page_chunks):
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                new_index_to_chunk[chunk_counter] = {
                    "text": chunk,
                    "document": doc["filename"],
                    "doc_id": doc["id"],
                    "chunk_index": chunk_counter,
                    "page_number": page_num
                }
                chunk_counter += 1
                
    if embeddings:
        embeddings_array = np.vstack(embeddings).astype('float32')
        new_index.add(embeddings_array)
        
    metadata["total_chunks"] = len(new_index_to_chunk)
    faiss.write_index(new_index, INDEX_PATH)
    with open(CHUNK_MAPPING_PATH, "wb") as f:
        pickle.dump(new_index_to_chunk, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    return True, f"Reindexed {len(metadata['documents'])} documents with {chunk_counter} chunks"

def get_query_suggestions(prefix):
    prefix = prefix.lower()
    suggestions = set()
    for key in SEARCH_CACHE.keys():
        hist_q = key.split('_')[0]
        if hist_q.startswith(prefix) and hist_q != prefix:
            suggestions.add(hist_q)
    try:
        for syn in wordnet.synsets(prefix):
            for l in syn.lemmas():
                syn_word = l.name().replace('_', ' ')
                if syn_word.startswith(prefix) and syn_word != prefix:
                    suggestions.add(syn_word)
    except:
        pass
    
    return list(suggestions)[:10]

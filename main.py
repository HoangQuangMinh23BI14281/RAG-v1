import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scraper import scrape_vnexpress
from rag_pipeline import process_and_store, setup_rag

def print_slow(text, delay=0.015):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def main():
    print("Bot đang khởi động...")

    # Giai đoạn 1: Scrape dữ liệu
    print("Đang thu thập dữ liệu từ VnExpress...")
    start_time = time.time()
    filename = scrape_vnexpress()
    scrape_time = time.time() - start_time
    print(f" Thời gian thu thập dữ liệu: {scrape_time:.2f} giây")

    if not filename:
        print("Không thể thu thập dữ liệu. Thoát chương trình.")
        return

    # Giai đoạn 2: Xử lý & tạo vector store
    print("Đang xử lý và lưu dữ liệu vào vector store...")
    start_time = time.time()
    vectordb = process_and_store(filename)
    vector_time = time.time() - start_time
    print(f" Thời gian tạo vector store: {vector_time:.2f} giây")

    # Giai đoạn 3: Setup LLM + RAG
    qa_chain = setup_rag(vectordb)

    print("\n=== Bot Tin Tức VnExpress ===")
    print("Hỏi tôi bất kỳ câu hỏi nào về tin tức trong ngày!")
    print("Nhập 'thoat' để thoát.\n")

    while True:
        query = input("Câu hỏi của bạn: ").strip()
        if query.lower() == "thoat":
            print("Tạm biệt!")
            break

        if not query:
            print("Vui lòng nhập câu hỏi.")
            continue

        print("\nĐang xử lý câu hỏi...")
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        duration = time.time() - start_time

        answer = result.get("result", "Không có kết quả")
        sources = result.get("source_documents", [])

        print("\n=== Trả lời ===")
        print_slow(answer)
        print(f"\n Thời gian xử lý câu hỏi: {duration:.2f} giây")

        print("\n=== Nguồn ===")
        for doc in sources:
            print(doc.metadata.get("source", "Không tìm thấy nguồn"))
        print()

if __name__ == "__main__":
    main()

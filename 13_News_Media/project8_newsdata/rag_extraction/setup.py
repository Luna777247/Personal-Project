#!/usr/bin/env python3
"""
Setup script for RAG Disaster Extraction System
Handles installation, configuration, and initial setup
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'chromadb',
        'sentence-transformers',
        'langchain',
        'openai',
        'anthropic',
        'groq',
        'pandas',
        'numpy',
        'python-dotenv',
        'tqdm',
        'colorama'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_vector_databases():
    """Check vector database availability"""
    print("\n🔍 Checking vector databases...")

    # Chroma - always available
    try:
        import chromadb
        print("✅ ChromaDB")
    except ImportError:
        print("❌ ChromaDB")

    # Qdrant
    try:
        import qdrant_client
        print("✅ Qdrant Client")
    except ImportError:
        print("⚠️  Qdrant Client (optional)")

    # Milvus
    try:
        import pymilvus
        print("✅ Milvus Client")
    except ImportError:
        print("⚠️  Milvus Client (optional)")


def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'config',
        'logs',
        'cache',
        'output'
    ]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")


def create_sample_config():
    """Create sample configuration files"""
    # Sample environment file
    env_content = """# API Keys for LLM providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here

# Vector Database URLs (optional)
QDRANT_URL=http://localhost:6333
MILVUS_HOST=localhost
MILVUS_PORT=19530

# System settings
LOG_LEVEL=INFO
CACHE_SIZE=1000
"""

    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("📝 Created .env file with sample configuration")
    else:
        print("⚠️  .env file already exists")


def create_sample_data():
    """Create sample disaster data for testing"""
    sample_data = [
        {
            "id": "sample_1",
            "content": """Bão số 12 gây thiệt hại nặng nề tại Quảng Nam. Theo báo cáo sơ bộ từ UBND tỉnh Quảng Nam, bão đã làm 3 người chết, 12 người bị thương. Hàng trăm ngôi nhà bị tốc mái, nhiều diện tích lúa và hoa màu bị ngập úng. Bộ Quốc phòng đã điều động lực lượng cứu hộ đến hỗ trợ.""",
            "metadata": {
                "source": "vnexpress",
                "date": "2023-11-15",
                "title": "Bão số 12: Quảng Nam thiệt hại nặng nề",
                "url": "https://vnexpress.net/bao-so-12-quang-nam-thiet-hai-nang-ne"
            }
        },
        {
            "id": "sample_2",
            "content": """Lũ lụt tại miền Trung Việt Nam. Mưa lớn kéo dài nhiều ngày đã gây ngập lụt nghiêm trọng tại các tỉnh Hà Tĩnh, Quảng Bình, Quảng Trị. Hàng nghìn hộ dân bị ảnh hưởng, nhiều tuyến đường bị chia cắt. Chính phủ đã chỉ đạo các bộ ngành hỗ trợ cứu trợ khẩn cấp.""",
            "metadata": {
                "source": "tuoitre",
                "date": "2023-10-20",
                "title": "Lũ lụt miền Trung: Hàng nghìn hộ dân bị ảnh hưởng",
                "url": "https://tuoitre.vn/lu-lut-mien-trung-hang-nghin-ho-dan-bi-anh-huong"
            }
        },
        {
            "id": "sample_3",
            "content": """Động đất tại Kon Tum. Trận động đất có độ lớn 5.1 richter xảy ra vào sáng nay tại huyện Kon Plông, tỉnh Kon Tum. Không có thiệt hại về người nhưng nhiều ngôi nhà bị nứt tường. Cơ quan chức năng đang đánh giá mức độ thiệt hại.""",
            "metadata": {
                "source": "dantri",
                "date": "2023-09-05",
                "title": "Động đất tại Kon Tum, không có thiệt hại về người",
                "url": "https://dantri.com.vn/dong-dat-tai-kon-tum-khong-co-thiet-hai-ve-nguoi"
            }
        }
    ]

    sample_file = Path('data/sample_disaster_data.json')
    if not sample_file.exists():
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print("📄 Created sample disaster data: data/sample_disaster_data.json")
    else:
        print("⚠️  Sample data file already exists")


def create_sample_queries():
    """Create sample extraction queries"""
    queries = [
        "Thiệt hại do bão số 12 tại Quảng Nam",
        "Tình hình lũ lụt tại miền Trung",
        "Động đất tại Kon Tum ngày 5/9/2023",
        "Số người chết trong các thảm họa gần đây",
        "Các tổ chức tham gia cứu trợ thảm họa"
    ]

    query_file = Path('data/sample_queries.txt')
    if not query_file.exists():
        with open(query_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(queries))
        print("📝 Created sample queries: data/sample_queries.txt")
    else:
        print("⚠️  Sample queries file already exists")


def run_initial_test():
    """Run initial system test"""
    print("\n🧪 Running initial system test...")

    try:
        from scripts.rag_extractor import create_rag_extractor

        # Create extractor
        extractor = create_rag_extractor()
        print("✅ RAG extractor created successfully")

        # Test metrics
        metrics = extractor.get_metrics()
        print(f"✅ System metrics: {metrics['total_documents']} documents, {metrics['total_chunks']} chunks")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def show_usage_guide():
    """Show usage guide"""
    print("\n" + "="*60)
    print("🚀 RAG DISASTER EXTRACTION SYSTEM - SETUP COMPLETE")
    print("="*60)

    print("\n📚 QUICK START GUIDE:")
    print("1. Configure API keys in .env file")
    print("2. Add your disaster data:")
    print("   python run_rag.py add --input data/sample_disaster_data.json")
    print("3. Search for information:")
    print("   python run_rag.py search --query 'bão tại Quảng Nam'")
    print("4. Extract disaster information:")
    print("   python run_rag.py extract --query 'thiệt hại bão số 12'")
    print("5. Run full demo:")
    print("   python scripts/demo_rag_extraction.py")

    print("\n📁 IMPORTANT FILES:")
    print("- run_rag.py: Main CLI interface")
    print("- scripts/rag_extractor.py: Core RAG engine")
    print("- config/rag_config.py: System configuration")
    print("- data/: Your data files")
    print("- output/: Extraction results")

    print("\n🔧 CONFIGURATION:")
    print("- Edit .env for API keys")
    print("- Modify config/rag_config.py for system settings")
    print("- Check README.md for detailed documentation")

    print("\n📊 MONITORING:")
    print("- python run_rag.py metrics  # View system metrics")
    print("- Check logs/ directory for detailed logs")

    print("\n" + "="*60)


def main():
    """Main setup function"""
    print("🚀 RAG DISASTER EXTRACTION SYSTEM SETUP")
    print("="*50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check vector databases
    check_vector_databases()

    # Create directories
    print("\n📁 Creating directories...")
    create_directories()

    # Create configuration files
    print("\n⚙️  Creating configuration...")
    create_sample_config()

    # Create sample data
    print("\n📄 Creating sample data...")
    create_sample_data()
    create_sample_queries()

    # Run initial test
    if run_initial_test():
        print("\n✅ Setup completed successfully!")
        show_usage_guide()
    else:
        print("\n❌ Setup completed with warnings. Please check configuration.")
        show_usage_guide()


if __name__ == "__main__":
    main()
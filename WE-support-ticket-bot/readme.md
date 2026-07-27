# WE Support Ticket Bot 🚀

An end-to-end multilingual natural language processing pipeline built during my internship at **WE (Telecom Egypt)**. Designed to automate customer support workflows by handling high volumes of tickets written in English, Arabic, and Franco-Arabic.

---

## 🌟 Key Features
* **Multi-Department Classification:** Automatically routes incoming tickets into *Billing, Network, Technical Support, Sales/Plans, or Other*.
* **Sentiment Analysis:** Detects Positive, Neutral, or Negative sentiment to flag frustrated customers for priority handling.
* **Abstractive Summarization:** Generates concise summaries so support agents can understand core issues instantly.
* **Interactive UI:** A Streamlit-powered dashboard supporting both real-time single-ticket evaluation and batch CSV processing.

---

## 🛠️ Tech Stack & Architecture
* **Models:** 
  * **MARBERTv2:** Fine-tuned for dialect-heavy classification and sentiment detection.
  * **mT5:** Fine-tuned for multilingual abstractive text summarization.
* **Frameworks:** PyTorch, Hugging Face Transformers, Streamlit.
* **Data:** Custom-generated synthetic dataset (~3,000 tickets) incorporating realistic Egyptian dialect variations, Franco-Arabic transliterations, typos, and emojis.

---

## 📂 Repository Structure
* `scripts/`: Contains the training and fine-tuning scripts for MARBERTv2 and mT5.
* `we_ticket_bot_app.py`: The main Streamlit application script.

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/Ahmed-Shehab-Source/ai-engineering-projects.git](https://github.com/Ahmed-Shehab-Source/ai-engineering-projects.git)
cd WE-support-ticket-bot

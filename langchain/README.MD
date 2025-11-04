# 🤖 Ollama: Run Large Language Models Locally

---

**Ollama** is the easiest way to run **Large Language Models (LLMs)** like Llama 3.1, Phi-3, and many others directly on your local machine.

It simplifies the process of getting up and running with AI models, acting much like a **containerization tool** such as Docker. Instead of managing complex dependencies, Ollama lets you download and run various open-source LLMs from a central repository with simple commands.

## ✨ Key Features and Benefits

| Feature | Description |
| :--- | :--- |
| **Local Execution** | Run LLMs on your computer for **enhanced privacy and security** by keeping data local. |
| **Offline Access** | Use the models even **without an internet connection** after the initial download. |
| **Easy Management** | Download, update, and delete models with simple **Command Line Interface (CLI) commands**. |
| **Cross-Platform** | Supports **macOS, Linux, and Windows**. |
| **GPU Acceleration** | Automatically uses your **GPU** to speed up processing when available, offering faster results. |
| **Simple API** | Includes a **REST API** for developers to integrate AI models into applications and workflows. |

---

## 🛠️ Installation and Setup

### 1. Install Ollama

Get started by installing the Ollama application.

➡️ **Download and follow the instructions on the official website:**
[https://ollama.ai](https://ollama.ai)

### 2. Pull and Run an LLM (Model Management)

Use the `pull` command in your terminal to download a model, and the `run` command to start an interactive chat session.

**Example: Pulling and running the `llama3.1` model.**

```bash
ollama pull llama3.1
ollama run llama3.1
```
Make sure the model supports tools.

Tool supports enables a model to answer a given prompt using tool(s) it knows about, making it possible for models to perform more complex tasks or interact with the outside world. Example tools include:

Functions and APIs

Web browsing

Code interpreter

https://ollama.com/blog/tool-support
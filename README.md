# Contract Q&A RAG System

## Project Overview

The goal of this project is to build a Contract Q&A Retrieval-Augmented Generation (RAG) system similar to Lizzy AI, an artificial contract assistant developed by an Israeli startup. Lizzy AI can draft and review contracts quickly, providing functionalities such as summarization, error detection, clause generation, and more. This challenge aims to create, evaluate, and improve a RAG system that allows users to interact with contracts and ask questions about them.

## Features

- **Interactive Q&A:** Allow users to chat with the system and ask questions about the contract.

## Technologies Used

- **LangChain:** Framework for building applications with LLMs.
- **Chroma:** Vector database for efficient data storage and retrieval.
- **OpenAI API:** Provides access to GPT-3.5-Turbo for language processing.
- **GPT-4:** Used for advanced language understanding and generation.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Contributing](#contributing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Dockerization](#dockerization)
- [Makefile](#makefile)
- [Documentation](#documentation)

## Introduction

Users can interact with the system to ask questions and get detailed answers about specific contracts. The system uses LangChain, Chroma, OpenAI API, and GPT-3.5-Turbo and GPT-4o-mini models.

## Project Structure

- `app.py`: The main Streamlit application.
- `Dockerfile`: Docker configuration for building the application container.
- `Makefile`: Automation of common tasks such as installation, build, and deployment.
- `.github/workflows/ci-cd.yml`: GitHub Actions configuration for CI/CD pipeline.
- `scripts`: Python scripts for various langchain retrievers and ragas evaluation.
- `notebooks`: All the notebooks.


## Getting Started

### Prerequisites

- **Python 3.7 or higher**

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cheronodaisy/contract-advisor-rag.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd contract-advisor-rag
   ```

3. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

4. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Build and run the Docker container:**

   - To build the Docker image:

     ```bash
     make build
     ```

   - To deploy the Docker container:

     ```bash
     make deploy
     ```

   Alternatively, you can run the Streamlit app directly without Docker

   ```bash
   streamlit run app.py
   ```

## Contributing

Contributions from the community are welcome. If you would like to contribute to this project, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b feature-branch
   ```

3. **Make your changes and commit them:**

   ```bash
   git commit -m 'Add some feature'
   ```

4. **Push to the branch:**

   ```bash
   git push origin feature-branch
   ```

5. **Open a pull request.**

## CI/CD Pipeline

- **GitHub Actions:** Automated workflows for building, testing, and deploying the application. Configured in the `.github/workflows/ci-cd.yml` file.

## Dockerization

- **Dockerfile:** Defines how the application is containerized and run.

## Makefile

- **Makefile:** Automates common tasks such as installing dependencies, building Docker images, and deploying containers. Use `make` commands to streamline the development workflow.

## Documentation

- **README:** This file includes instructions for setting up the development environment, running the project, and utilizing the implemented tools.
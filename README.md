# Disease-outbreak-predictor
"Machine learning–based system that predicts disease outbreak regions across India and visualizes them on an interactive map. Integrated with an AI chatbot that answers user queries about diseases, symptoms, and prevention, enabling data-driven health awareness and early intervention."

This project presents an integrated system that combines machine learning and conversational AI to predict potential disease outbreak regions across India and provide relevant disease-related insights through a chatbot interface. Developed using Flask, the system uses multiple machine learning algorithms including CatBoost, Logistic Regression, Support Vector Machine, and Gradient Boosting to analyze climatic, health, and environmental parameters and identify areas at high risk of outbreaks. The final implementation employs the CatBoost Classifier, chosen for its superior accuracy and ability to handle complex, multi-class data efficiently. It gave an accuracy of 86%.

The model processes parameters such as temperature, precipitation, and Leaf Area Index (LAI), along with temporal lag features, to predict disease probabilities for each district. The predictions are visualized on an interactive map of India using Folium, where regions are dynamically highlighted based on outbreak intensity. This visualization allows users to easily interpret and monitor disease risk patterns geographically.

An AI-driven research chatbot is integrated into the application to enhance accessibility and user engagement. Built using LangChain, HuggingFace embeddings, and the Mistral API, the chatbot retrieves information from stored PDF and PowerPoint documents to answer user queries about diseases, symptoms, and prevention strategies. It provides concise, context-aware responses derived from available research data, making it a useful tool for health information and awareness.

The project demonstrates how data science, machine learning, and natural language processing can be effectively combined to address real-world public health challenges. It also provides practical features such as disease-specific precaution suggestions and an intuitive web interface. By visualizing predictive insights and enabling natural interactions, the system supports proactive health management and data-driven decision-making.

Running the Application:
Install dependencies:
pip install -r requirements.txt

Add your MISTRAL_API_KEY in a .env file:
MISTRAL_API_KEY=your_api_key_here

Place your data files in:
mlproject.xlsx for model input

Data/ folder for chatbot PDFs or PPTX documents

Run the app:
python app.py

Open in browser:
http://127.0.0.1:5000/

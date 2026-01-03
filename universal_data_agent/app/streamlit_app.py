import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import contextlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title(" Universal Data Analysis Agent")

uploaded_file = st.file_uploader("Upload a CSV file (max 1GB)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(" Dataset loaded successfully!")

    st.write("### Full Dataset")
    st.dataframe(df)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    chart_type = st.selectbox(
        " Select Graph Type",
        ["Auto (AI Decide)", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"]
    )

    user_query = st.text_input("Ask a question about your dataset:")

    if user_query:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful data assistant. 
Always answer using the uploaded dataset (df).
When generating charts:
- ALWAYS use matplotlib (plt)
- ALWAYS end with st.pyplot(plt)
- Use only selected graph type if provided.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Dataset columns: {list(df.columns)}
Chart Type Selected: {chart_type}
User Question: {user_query}

Write Python code using pandas + matplotlib.
Code MUST show table using st.dataframe and graph using st.pyplot(plt).
Use variable df only.
"""
                }
            ]
        )

        code = response.choices[0].message.content.strip()

        if "```" in code:
            parts = code.split("```")
            for p in parts:
                if p.strip().startswith("python"):
                    code = p.replace("python", "", 1).strip()
                    break
                elif "import" in p:
                    code = p.strip()
                    break

        st.markdown("###  Generated Python Code")
        st.code(code, language="python")

        local_vars = {"df": df, "st": st, "plt": plt, "pd": pd}

        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code, {}, local_vars)
            output = f.getvalue()
            if output:
                st.text(output)
        except Exception as e:
            st.error(f" Error executing generated code: {e}")


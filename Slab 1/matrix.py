import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Matrix Operations Tool", layout="wide")

st.title("🔢 Matrix Operations Tool")
st.write("Perform basic matrix operations using NumPy")

# -----------------------
# Helper functions
# -----------------------
def parse_matrix(input_text):
    """Convert user input into numpy array."""
    try:
        rows = input_text.strip().split("\n")
        matrix = [list(map(float, row.split())) for row in rows if row.strip()]
        return np.array(matrix)
    except Exception:
        return None

def download_button(matrix, label, filename):
    """Create a download button for a matrix (CSV or Excel)."""
    df = pd.DataFrame(matrix)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=f"💾 Download {label} (CSV)", data=csv,
                       file_name=f"{filename}.csv", mime="text/csv")

    excel = df.to_excel(f"{filename}.xlsx", index=False, engine="openpyxl")
    with open(f"{filename}.xlsx", "rb") as f:
        st.download_button(label=f"💾 Download {label} (Excel)",
                           data=f, file_name=f"{filename}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------
# Matrix Input Section
# -----------------------
st.sidebar.header("⚙️ Matrix Input Options")
input_mode = st.sidebar.radio("How do you want to input matrices?",
                              ["Manual Entry", "Random Generator"])

if input_mode == "Manual Entry":
    st.subheader("Enter Matrix A")
    matrix_a_input = st.text_area("Matrix A (rows separated by newline, values by space)", "1 2 3\n4 5 6")

    st.subheader("Enter Matrix B")
    matrix_b_input = st.text_area("Matrix B (rows separated by newline, values by space)", "7 8 9\n10 11 12")

    matrix_a = parse_matrix(matrix_a_input)
    matrix_b = parse_matrix(matrix_b_input)

else:  # Random Generator
    st.subheader("🎲 Random Matrix Generator")
    rows_a = st.number_input("Rows in Matrix A", 1, 10, 2)
    cols_a = st.number_input("Columns in Matrix A", 1, 10, 2)
    rows_b = st.number_input("Rows in Matrix B", 1, 10, 2)
    cols_b = st.number_input("Columns in Matrix B", 1, 10, 2)

    if st.button("Generate Random Matrices"):
        matrix_a = np.random.randint(1, 10, size=(rows_a, cols_a))
        matrix_b = np.random.randint(1, 10, size=(rows_b, cols_b))
    else:
        matrix_a, matrix_b = None, None

# -----------------------
# Display matrices
# -----------------------
if matrix_a is not None:
    st.write("### Matrix A")
    st.write(matrix_a)

if matrix_b is not None:
    st.write("### Matrix B")
    st.write(matrix_b)

# -----------------------
# Operations Section (Tabs for UI)
# -----------------------
if matrix_a is not None and matrix_b is not None:
    st.subheader("📊 Matrix Operations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["➕ Addition", "➖ Subtraction", "✖️ Multiplication", "🔄 Transpose", "|A| Determinant"]
    )

    with tab1:
        st.header("Addition (A + B)")
        if matrix_a.shape == matrix_b.shape:
            result = matrix_a + matrix_b
            st.write(result)
            download_button(result, "A+B", "addition_result")
        else:
            st.error("⚠️ Matrices must have the same dimensions for addition.")

    with tab2:
        st.header("Subtraction (A - B)")
        if matrix_a.shape == matrix_b.shape:
            result = matrix_a - matrix_b
            st.write(result)
            download_button(result, "A-B", "subtraction_result")
        else:
            st.error("⚠️ Matrices must have the same dimensions for subtraction.")

    with tab3:
        st.header("Multiplication (A × B)")
        if matrix_a.shape[1] == matrix_b.shape[0]:
            result = np.matmul(matrix_a, matrix_b)
            st.write(result)
            download_button(result, "A×B", "multiplication_result")
        else:
            st.error("⚠️ Number of columns in A must equal rows in B for multiplication.")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transpose of A")
            result = np.transpose(matrix_a)
            st.write(result)
            download_button(result, "Transpose A", "transpose_a")
        with col2:
            st.subheader("Transpose of B")
            result = np.transpose(matrix_b)
            st.write(result)
            download_button(result, "Transpose B", "transpose_b")

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            if matrix_a.shape[0] == matrix_a.shape[1]:
                st.subheader("Determinant of A")
                result = np.linalg.det(matrix_a)
                st.write(result)
            else:
                st.warning("Matrix A must be square for determinant.")
        with col2:
            if matrix_b.shape[0] == matrix_b.shape[1]:
                st.subheader("Determinant of B")
                result = np.linalg.det(matrix_b)
                st.write(result)
            else:
                st.warning("Matrix B must be square for determinant.")
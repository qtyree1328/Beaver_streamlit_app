import streamlit as st
st.set_page_config(layout="wide")

# Place the logo at the top
st.sidebar.image("assets/pngimg.com - beaver_PNG35.png", use_column_width=True)

about_page = st.Page(
    page = "pages/About_Lab.py",
    title = "About Me!",
    icon = ":material/account_circle:",
    default = True,
)

project_1_page = st.Page(
    page = "pages/Exports_page.py",
    title = "Export Images!",
    icon = ":material/bar_chart:",

)

pg = st.navigation(pages= [about_page,project_1_page])
# st.logo("assets\pngimg.com - beaver_PNG35.png")
pg.run()
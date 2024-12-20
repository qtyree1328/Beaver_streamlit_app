import streamlit as st

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

pg.run()
import streamlit as st


from Home import face_rec

st.set_page_config(page_title='THMC attendance report', layout='centered')
st.subheader("THMC Report")

name = "attendance:logs"

def read_logs(name, end=-1):
    logs = face_rec.r.lrange(name, 0, end)
    return logs

tab1, tab2 = st.tabs(["Registered members","Attendance Logs"])

with tab1:
    with st.spinner("Retrieving data from database..."):
        redis_face_db = face_rec.retrive_data(name = "academy:register")
        st.dataframe(redis_face_db, use_container_width=True)

with tab2:
    if st.button("Read Attendance Logs"):    
        with st.spinner("Retrieving data from database..."):    
            st.write(read_logs(name))



# if st.button("Clear Attendance Logs"):
#     face_rec.r.delete(name)
#     st.success("Attendance logs cleared successfully.")
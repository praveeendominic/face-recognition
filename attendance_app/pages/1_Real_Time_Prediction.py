import streamlit as st

from Home import face_rec

st.set_page_config(page_title='THMC Face recognition System', layout='centered')
st.subheader('Real-time Face recognition')

obj = face_rec.RealTimePred()

# Retrieve data from DB
with st.spinner("Retrieving data from Redis db ..."):
    redis_face_db = face_rec.retrive_data(name = "academy:register")
    st.dataframe(redis_face_db, use_container_width=True)
st.success('Data retrieved successfully from Redis')

from streamlit_webrtc import webrtc_streamer
import av
import time



def video_frame_callback(frame):
    # global setTime
    
    img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
    # operation that you can perform on the array

    try:
        pred_img = obj.face_prediction(img, redis_face_db,
                                            'facial_features',
                                            ['Name', 'Role'],
                                            thresh=0.5)
        st.text("Real-time face recognition in progress...")
        print("Real-time face recognition in progress...")

    except Exception as e:
        st.error(f"Error during face prediction: {e}")
        print.error(f"Error during face prediction: {e}")
              
    # 
    # timenow = time.time()
    # difftime = timenow - setTime
    # if difftime >= waitTime:
    #     realtimepred.saveLogs_redis()
    #     setTime = time.time() # reset time        
    #     print('Save Data to redis database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")

#     flipped = img[::-1,:,:]

#     return av.VideoFrame.from_ndarray(flipped, format="bgr24")

# webrtc_streamer(key="example", video_frame_callback=video_frame_callback)


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
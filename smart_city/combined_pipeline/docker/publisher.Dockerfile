FROM jrottenberg/ffmpeg:6-ubuntu
COPY publish_input_rtsp.sh /scripts/publish_input_rtsp.sh
RUN chmod +x /scripts/publish_input_rtsp.sh
ENTRYPOINT ["/bin/bash", "/scripts/publish_input_rtsp.sh"]

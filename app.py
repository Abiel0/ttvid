import os
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gradio_client import Client
import shutil
import logging

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def serve_frontend():
    return send_from_directory(current_dir, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    try:
        data = request.json
        prompt = data.get('prompt', 'Hello!!')
        
        app.logger.info(f"Received prompt: {prompt}")
        
        client = Client("TIGER-Lab/T2V-Turbo-V2")
        result = client.predict(
            prompt=prompt,
            guidance_scale=7.5,
            percentage=0.5,
            num_inference_steps=16,
            num_frames=16,
            seed=1968164510,
            randomize_seed=True,
            param_dtype="bf16",
            api_name="/predict"
        )

        app.logger.info(f"API result: {result}")

        if result is not None and isinstance(result, tuple) and len(result) > 0:
            result_dict = result[0]
            
            if isinstance(result_dict, dict) and 'video' in result_dict:
                temp_video_path = result_dict['video']
                new_video_path = os.path.join(current_dir, "generated_video.mp4")
                
                if os.path.exists(temp_video_path):
                    shutil.copy2(temp_video_path, new_video_path)
                    return jsonify({"message": "Video generated successfully", "video_path": "/generated_video.mp4"})
                else:
                    app.logger.error(f"Generated video file not found at {temp_video_path}")
                    return jsonify({"error": "Generated video file not found"}), 404
            else:
                app.logger.error("No video data in API response")
                return jsonify({"error": "No video data in API response"}), 500
        else:
            app.logger.error(f"Unexpected API response format: {result}")
            return jsonify({"error": "Unexpected API response format"}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/generated_video.mp4')
def serve_video():
    return send_from_directory(current_dir, 'generated_video.mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
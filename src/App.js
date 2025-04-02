import { useState, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  const processAudioData = async (audioData) => {
    try {
      // Create inference session
      const session = await ort.InferenceSession.create('./my_audio_classifier_2.onnx');

      // Create tensor from audio data
      const inputTensor = new ort.Tensor(
        'float32',
        new Float32Array(audioData),
        [1, 1, 16000]
      );

      // Prepare feeds
      const feeds = {};
      feeds[session.inputNames[0]] = inputTensor;

      // Run the model
      const results = await session.run(feeds);
      const outputTensor = Object.values(results)[0];
      
      // Get all three prediction values (noise, inhale, exhale)
      const predictions = {
        noise: outputTensor.data[0],
        inhale: outputTensor.data[1],
        exhale: outputTensor.data[2]
      };
      
      setPrediction(predictions);
    } catch (error) {
      console.error('Error running the model:', error);
    }
  };

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(16384, 1, 1);
      
      let audioData = [];
      const sampleRate = 16000;
      const duration = 1; // Changed from 4 to 1 second
      const requiredSamples = sampleRate * duration; // Now equals 16000
      
      setIsRecording(true);
      console.log(audioContext.sampleRate)

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Downsample to 16kHz if needed
        if (audioContext.sampleRate !== sampleRate) {
          const ratio = audioContext.sampleRate / sampleRate;
          for (let i = 0; i < inputData.length; i += ratio) {
            audioData.push(inputData[Math.floor(i)]);
          }
        } else {
          audioData = audioData.concat(Array.from(inputData));
        }

        // Process every 16000 samples (1 second) and keep the most recent data
        if (audioData.length >= requiredSamples) {
          const finalAudioData = audioData.slice(-requiredSamples);
          processAudioData(finalAudioData);
          // Keep only the most recent 0.75 seconds of data
          audioData = audioData.slice(-((requiredSamples * 3) / 4));
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    // Add cleanup code here if needed
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>Audio Classifier Test</p>
        <button 
          onClick={isRecording ? stopRecording : startRecording}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
        {prediction !== null && (
          <>
            <p>Predictions:</p>
            <ul>
              <li>Noise: {prediction.noise.toFixed(4)}</li>
              <li>Inhale: {prediction.inhale.toFixed(4)}</li>
              <li>Exhale: {prediction.exhale.toFixed(4)}</li>
            </ul>
            <p>Detected: {
              Math.max(prediction.noise, prediction.inhale, prediction.exhale) === prediction.noise 
                ? 'Noise' 
                : Math.max(prediction.inhale, prediction.exhale) === prediction.inhale 
                  ? 'Inhale' 
                  : 'Exhale'
            }</p>
          </>
        )}
      </header>
    </div>
  );
}

export default App;

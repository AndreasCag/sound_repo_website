import { useState, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  const processAudioData = async (audioData) => {
    try {
      // Create inference session
      const session = await ort.InferenceSession.create('./my_audio_classifier.onnx');

      // Create tensor from audio data
      const inputTensor = new ort.Tensor(
        'float32',
        new Float32Array(audioData),
        [1, 1, 64000]
      );

      // Prepare feeds
      const feeds = {};
      feeds[session.inputNames[0]] = inputTensor;

      // Run the model
      const results = await session.run(feeds);
      const outputTensor = Object.values(results)[0];
      const predictionValue = outputTensor.data[0];
      setPrediction(predictionValue);
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
      const duration = 4; // seconds
      const requiredSamples = sampleRate * duration;
      
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

        // Process every 64000 samples (4 seconds) and keep the most recent data
        if (audioData.length >= requiredSamples) {
          const finalAudioData = audioData.slice(-requiredSamples);
          processAudioData(finalAudioData);
          // Keep only the most recent 3 seconds of data
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
            <p>Prediction: {prediction.toFixed(4) > 0 ? 'Breathing' : 'Not breathing'}</p>
            <p>{prediction.toFixed(4)}</p>
          </>
        )}
      </header>
    </div>
  );
}

export default App;

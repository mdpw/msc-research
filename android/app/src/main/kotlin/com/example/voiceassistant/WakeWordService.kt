package com.example.voiceassistant

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.sqrt

class WakeWordService(private val context: Context) {
    private val TAG = "WakeWordService"

    // Audio configuration
    private val SAMPLE_RATE = 16000
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)

    // Model configuration
    private val MODEL_INPUT_SAMPLES = 40
    private val WINDOW_SIZE_SAMPLES = 640

    // Wake word detection parameters - VERY SENSITIVE
    private val CONFIDENCE_THRESHOLD = 0.32f  // Lowered significantly for sensitivity
    private val ENERGY_THRESHOLD = 0.02f      // Lower energy requirement
    private val MIN_ZERO_CROSSING_RATE = 0.04f

    private var audioRecord: AudioRecord? = null
    private var interpreter: Interpreter? = null
    private var isListening = false
    private var detectionThread: Thread? = null

    private var onDetectedCallback: ((String) -> Unit)? = null
    private var onErrorCallback: ((String) -> Unit)? = null

    fun initialize() {
        try {
            Log.d(TAG, "Initializing WakeWordService...")

            val modelPath = "models/wake_word/wake_word_model.tflite"
            val model = loadModelFile(modelPath)

            val options = Interpreter.Options().apply {
                setNumThreads(2)
                setUseNNAPI(false)
            }

            interpreter = Interpreter(model, options)

            val inputTensor = interpreter?.getInputTensor(0)
            Log.d(TAG, "✅ Model loaded - Input shape: ${inputTensor?.shape()?.contentToString()}, dtype: ${inputTensor?.dataType()}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Initialization error", e)
            onErrorCallback?.invoke("Failed to initialize: ${e.message}")
        }
    }

    private fun loadModelFile(path: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun startListening(
        onDetected: (String) -> Unit,
        onErrorCallback: (String) -> Unit
    ) {
        if (isListening) {
            Log.w(TAG, "⚠️ Already listening")
            return
        }

        this.onDetectedCallback = onDetected
        this.onErrorCallback = onErrorCallback

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE * 4
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                throw Exception("AudioRecord not initialized")
            }

            audioRecord?.startRecording()
            isListening = true

            detectionThread = thread {
                detectWakeWord()
            }

            Log.d(TAG, "✅ Started listening for wake words (threshold: $CONFIDENCE_THRESHOLD)")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error starting listening", e)
            this.onErrorCallback?.invoke("Failed to start: ${e.message}")
        }
    }

    fun stopListening() {
        Log.d(TAG, "🛑 Stopping wake word detection...")
        isListening = false

        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            detectionThread?.interrupt()
            detectionThread = null

            Log.d(TAG, "✅ Wake word detection stopped")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error stopping", e)
        }
    }

    private fun detectWakeWord() {
        val audioBuffer = ShortArray(WINDOW_SIZE_SAMPLES)
        val floatBuffer = FloatArray(WINDOW_SIZE_SAMPLES)

        while (isListening) {
            try {
                val readSize = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0

                if (readSize <= 0) {
                    Thread.sleep(10)
                    continue
                }

                // Convert to float and normalize
                for (i in 0 until readSize) {
                    floatBuffer[i] = audioBuffer[i].toFloat() / 32768.0f
                }

                // Calculate audio features
                val energy = calculateEnergy(floatBuffer, readSize)
                val zcr = calculateZeroCrossingRate(floatBuffer, readSize)

                // Basic audio quality check
                if (energy < ENERGY_THRESHOLD || zcr < MIN_ZERO_CROSSING_RATE) {
                    Thread.sleep(50)
                    continue
                }

                // Downsample to MODEL_INPUT_SAMPLES
                val downsampledAudio = downsampleAudio(floatBuffer, readSize, MODEL_INPUT_SAMPLES)

                // Run inference
                val predictions = runInference(downsampledAudio)

                if (predictions != null) {
                    val bgConfidence = predictions[0]
                    val helloConfidence = predictions[1]
                    val hiConfidence = predictions[2]

                    val maxWakeWordConf = maxOf(helloConfidence, hiConfidence)

                    // Check if wake word confidence exceeds threshold
                    if (maxWakeWordConf > CONFIDENCE_THRESHOLD) {
                        val detectedWord = if (helloConfidence > hiConfidence) "Hello Hotel" else "Hi Hotel"
                        Log.d(TAG, "🎯 WAKE WORD DETECTED: $detectedWord (conf: ${"%.3f".format(maxWakeWordConf)}) | bg=${"%.3f".format(bgConfidence)}, Energy=${"%.4f".format(energy)}")

                        onDetectedCallback?.invoke(detectedWord)

                        // Cooldown period
                        Thread.sleep(2000)
                    }
                }

                Thread.sleep(50)

            } catch (e: InterruptedException) {
                break
            } catch (e: Exception) {
                Log.e(TAG, "❌ Detection error", e)
            }
        }
    }

    private fun downsampleAudio(audio: FloatArray, size: Int, targetSize: Int): FloatArray {
        val downsampled = FloatArray(targetSize)
        val step = size.toFloat() / targetSize

        for (i in 0 until targetSize) {
            val index = (i * step).toInt()
            downsampled[i] = if (index < size) audio[index] else 0f
        }

        return downsampled
    }

    private fun runInference(audioData: FloatArray): FloatArray? {
        try {
            val inputBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SAMPLES * 4).apply {
                order(ByteOrder.nativeOrder())
                for (i in 0 until MODEL_INPUT_SAMPLES) {
                    putFloat(audioData[i])
                }
                rewind()
            }

            val outputBuffer = ByteBuffer.allocateDirect(3 * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            interpreter?.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val predictions = FloatArray(3)
            for (i in 0..2) {
                predictions[i] = outputBuffer.float
            }

            return predictions

        } catch (e: Exception) {
            Log.e(TAG, "❌ Inference error", e)
            return null
        }
    }

    private fun calculateEnergy(buffer: FloatArray, size: Int): Float {
        var sum = 0.0
        for (i in 0 until size) {
            sum += buffer[i] * buffer[i]
        }
        return sqrt(sum / size).toFloat()
    }

    private fun calculateZeroCrossingRate(buffer: FloatArray, size: Int): Float {
        var crossings = 0
        for (i in 1 until size) {
            if ((buffer[i] >= 0 && buffer[i - 1] < 0) || (buffer[i] < 0 && buffer[i - 1] >= 0)) {
                crossings++
            }
        }
        return crossings.toFloat() / size
    }

    fun release() {
        Log.d(TAG, "Releasing resources...")
        stopListening()
        interpreter?.close()
        interpreter = null
    }
}
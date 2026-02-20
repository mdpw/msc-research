package com.example.voiceassistant

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

class AudioRecorder(private val context: Context) {

    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val TAG = "AudioRecorder"

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val audioBuffer = mutableListOf<Float>()
    private var recordingThread: Thread? = null

    fun startRecording() {
        if (ActivityCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "❌ Audio permission not granted")
            throw SecurityException("Audio permission not granted")
        }

        try {
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            ) * 2

            Log.d(TAG, "📱 Creating AudioRecord with buffer size: $bufferSize")

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "❌ AudioRecord not initialized properly")
                throw IllegalStateException("AudioRecord initialization failed")
            }

            audioBuffer.clear()
            audioRecord?.startRecording()
            isRecording = true

            Log.d(TAG, "🎙️ Recording started")

            recordingThread = Thread {
                val buffer = ShortArray(bufferSize / 2)

                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0

                    if (read > 0) {
                        // Convert to float and store
                        for (i in 0 until read) {
                            audioBuffer.add(buffer[i] / 32768.0f)
                        }
                    } else {
                        Log.e(TAG, "⚠️ Read returned: $read")
                    }
                }
                Log.d(TAG, "📊 Recording thread finished")
            }
            recordingThread?.start()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error starting recording: ${e.message}", e)
            throw e
        }
    }

    fun stopRecording(): FloatArray {
        try {
            Log.d(TAG, "⏹️ Stopping recording...")
            isRecording = false

            // Wait for recording thread to finish
            recordingThread?.join(1000)

            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            val result = audioBuffer.toFloatArray()
            audioBuffer.clear()

            Log.d(TAG, "✅ Recording stopped - Captured ${result.size} samples (${result.size / SAMPLE_RATE.toFloat()} seconds)")

            return result

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error stopping recording: ${e.message}", e)
            return FloatArray(0)
        }
    }

    /**
     * Record with Voice Activity Detection.
     * Automatically stops after silence is detected following speech,
     * or after maxDurationMs as a hard cap.
     *
     * VAD state machine:
     *   WAITING_FOR_SPEECH → (energy > threshold) → SPEECH_ACTIVE
     *   SPEECH_ACTIVE → (energy < threshold for silenceTimeoutMs) → done
     *   Any state → (elapsed > maxDurationMs) → done
     */
    suspend fun recordWithVAD(
        silenceTimeoutMs: Long = 1500L,
        maxDurationMs: Long = 10000L,
        speechEnergyThreshold: Float = 0.02f,
        onStateChange: ((String) -> Unit)? = null
    ): FloatArray = withContext(Dispatchers.IO) {
        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            throw SecurityException("Audio permission not granted")
        }

        val result = CompletableDeferred<FloatArray>()

        try {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * 2

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                throw IllegalStateException("AudioRecord initialization failed")
            }

            val vadBuffer = mutableListOf<Float>()
            recorder.startRecording()
            Log.d(TAG, "🎙️ VAD recording started (silence=${silenceTimeoutMs}ms, max=${maxDurationMs}ms)")

            val readBuffer = ShortArray(bufferSize / 2)
            val startTime = System.currentTimeMillis()
            var speechDetected = false
            var silenceStartTime = 0L

            var running = true
            while (running) {
                val read = recorder.read(readBuffer, 0, readBuffer.size)
                if (read <= 0) continue

                // Store audio samples
                for (i in 0 until read) {
                    vadBuffer.add(readBuffer[i] / 32768.0f)
                }

                // Calculate RMS energy for this chunk
                var sum = 0.0
                for (i in 0 until read) {
                    val sample = readBuffer[i] / 32768.0f
                    sum += sample * sample
                }
                val rmsEnergy = sqrt(sum / read).toFloat()

                val elapsed = System.currentTimeMillis() - startTime

                // Hard cap on recording duration
                if (elapsed >= maxDurationMs) {
                    Log.d(TAG, "⏱️ Max duration reached (${maxDurationMs}ms)")
                    onStateChange?.invoke("Max time reached")
                    running = false
                    continue
                }

                if (rmsEnergy >= speechEnergyThreshold) {
                    // Speech detected
                    if (!speechDetected) {
                        speechDetected = true
                        Log.d(TAG, "🗣️ Speech detected at ${elapsed}ms (energy=${"%.4f".format(rmsEnergy)})")
                        onStateChange?.invoke("Listening...")
                    }
                    // Reset silence timer whenever we hear speech
                    silenceStartTime = 0L
                } else if (speechDetected) {
                    // Silence after speech
                    if (silenceStartTime == 0L) {
                        silenceStartTime = System.currentTimeMillis()
                    }
                    val silenceDuration = System.currentTimeMillis() - silenceStartTime
                    if (silenceDuration >= silenceTimeoutMs) {
                        Log.d(TAG, "🔇 Silence detected for ${silenceDuration}ms — stopping")
                        onStateChange?.invoke("Processing...")
                        running = false
                    }
                }
            }

            recorder.stop()
            recorder.release()

            val audioData = vadBuffer.toFloatArray()
            val durationSec = audioData.size / SAMPLE_RATE.toFloat()
            Log.d(TAG, "✅ VAD recording done - ${audioData.size} samples (${"%.1f".format(durationSec)}s), speechDetected=$speechDetected")

            result.complete(audioData)

        } catch (e: Exception) {
            Log.e(TAG, "❌ VAD recording error: ${e.message}", e)
            result.complete(FloatArray(0))
        }

        result.await()
    }
}
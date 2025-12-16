package com.example.voiceassistant

import android.content.Context
import android.util.Log
import org.vosk.Model
import org.vosk.Recognizer
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class VoskService(private val context: Context) {

    private var model: Model? = null
    private var recognizer: Recognizer? = null
    private val TAG = "VoskService"

    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            //val modelDir = File(context.filesDir, "vosk-model")
            val modelDir = File(context.filesDir, "vosk-model-en-us-0.22-lgraph")

            if (!modelDir.exists()) {
                Log.d(TAG, "📥 Copying model from assets...")
                copyAssetFolder("models", modelDir.absolutePath)
            } else {
                Log.d(TAG, "✅ Model already exists")
            }

            Log.d(TAG, "🔍 Initializing Vosk model...")
            model = Model(modelDir.absolutePath)

            // Create recognizer once during initialization
            recognizer = Recognizer(model, 16000.0f)

            Log.d(TAG, "✅ Vosk initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Init failed: ${e.message}", e)
            throw Exception("Failed to initialize Vosk: ${e.message}")
        }
    }

    private fun copyAssetFolder(srcName: String, dstName: String) {
        val assetManager = context.assets
        val files = assetManager.list(srcName) ?: arrayOf()

        val destDir = File(dstName)
        if (!destDir.exists()) {
            destDir.mkdirs()
        }

        for (filename in files) {
            val srcPath = "$srcName/$filename"
            val dstPath = "$dstName/$filename"

            val subFiles = assetManager.list(srcPath)
            if (subFiles != null && subFiles.isNotEmpty()) {
                copyAssetFolder(srcPath, dstPath)
            } else {
                assetManager.open(srcPath).use { input ->
                    File(dstPath).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    suspend fun transcribeAudio(audioData: FloatArray): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "🎤 Starting transcription of ${audioData.size} samples")

            if (recognizer == null) {
                Log.e(TAG, "❌ Recognizer is null!")
                return@withContext ""
            }

            if (audioData.isEmpty()) {
                Log.e(TAG, "❌ Audio data is empty!")
                return@withContext ""
            }

            // Reset recognizer for new audio
            recognizer?.reset()

            // Convert float to byte array (16-bit PCM)
            val byteArray = ByteArray(audioData.size * 2)
            for (i in audioData.indices) {
                val sample = (audioData[i] * 32767).toInt().coerceIn(-32768, 32767)
                byteArray[i * 2] = (sample and 0xFF).toByte()
                byteArray[i * 2 + 1] = ((sample shr 8) and 0xFF).toByte()
            }
            Log.d(TAG, "✅ Audio converted to bytes: ${byteArray.size} bytes")

            // Feed all audio at once
            recognizer?.acceptWaveForm(byteArray, byteArray.size)

            // Get final result
            val result = recognizer?.finalResult ?: ""
            Log.d(TAG, "✅ Got result: $result")

            val json = JSONObject(result)
            val text = json.optString("text", "")

            Log.d(TAG, "🎤 Transcribed: '$text'")
            return@withContext text.trim()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Transcription failed: ${e.message}", e)
            e.printStackTrace()
            return@withContext ""
        }
    }

    fun release() {
        try {
            recognizer?.close()
            recognizer = null
            model?.close()
            model = null
            Log.d(TAG, "🔚 Vosk released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing Vosk: ${e.message}", e)
        }
    }
}
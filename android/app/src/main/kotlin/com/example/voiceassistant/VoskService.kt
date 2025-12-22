package com.example.voiceassistant

import android.content.Context
import android.util.Log
import org.vosk.Model
import org.vosk.Recognizer
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

class VoskService(private val context: Context) {

    private var model: Model? = null
    private var recognizer: Recognizer? = null
    private val TAG = "VoskService"
    
    private val initMutex = Mutex()
    private var isReady = false

    suspend fun initialize() = withContext(Dispatchers.IO) {
        initMutex.withLock {
            if (isReady) return@withLock
            
            try {
                Log.d(TAG, "🔧 Starting Vosk initialization...")
                val modelDir = File(context.filesDir, "vosk-model-en-us-0.22")

                if (!modelDir.exists()) {
                    Log.d(TAG, "📥 Copying model from assets (this may take a while)...")
                    copyAssetFolder("models", modelDir.absolutePath)
                }

                Log.d(TAG, "🔍 Loading Vosk model into memory...")
                model = Model(modelDir.absolutePath)
                recognizer = Recognizer(model, 16000.0f)
                
                isReady = true
                Log.d(TAG, "✅ Vosk initialized successfully")

            } catch (e: Exception) {
                Log.e(TAG, "❌ Init failed: ${e.message}", e)
                throw Exception("Failed to initialize Vosk: ${e.message}")
            }
        }
    }

    private fun copyAssetFolder(srcName: String, dstName: String) {
        val assetManager = context.assets
        val files = assetManager.list(srcName) ?: arrayOf()

        val destDir = File(dstName)
        if (!destDir.exists()) destDir.mkdirs()

        for (filename in files) {
            val srcPath = if (srcName.isEmpty()) filename else "$srcName/$filename"
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
        // Wait if initialization is still happening
        if (!isReady) {
            Log.d(TAG, "⏳ Transcription requested but Vosk not ready. Waiting for init...")
            initialize() 
        }

        try {
            val rec = recognizer ?: return@withContext ""
            Log.d(TAG, "🎤 Transcribing ${audioData.size} samples...")

            // Reset recognizer for fresh start
            rec.reset()

            // Convert FloatArray to 16-bit PCM ByteArray
            val byteArray = ByteArray(audioData.size * 2)
            for (i in audioData.indices) {
                val sample = (audioData[i] * 32767).toInt().coerceIn(-32768, 32767)
                byteArray[i * 2] = (sample and 0xFF).toByte()
                byteArray[i * 2 + 1] = ((sample shr 8) and 0xFF).toByte()
            }

            // Feed in chunks of 4096 bytes for better processing
            val chunkSize = 4096
            var offset = 0
            while (offset < byteArray.size) {
                val length = Math.min(chunkSize, byteArray.size - offset)
                rec.acceptWaveForm(byteArray.sliceArray(offset until offset + length), length)
                offset += length
            }

            val finalResult = rec.finalResult
            Log.d(TAG, "✅ Raw Result: $finalResult")

            val text = JSONObject(finalResult).optString("text", "")
            Log.d(TAG, "🎤 Final Transcription: '$text'")
            
            return@withContext text.trim()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Transcription error: ${e.message}")
            return@withContext ""
        }
    }

    fun release() {
        try {
            recognizer?.close()
            model?.close()
            isReady = false
            Log.d(TAG, "🔚 Vosk resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing Vosk: ${e.message}")
        }
    }
}
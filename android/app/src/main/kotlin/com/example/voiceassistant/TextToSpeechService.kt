package com.example.voiceassistant

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.*

/**
 * Text-to-Speech Service
 * Provides voice responses to guests
 */
class TextToSpeechService(private val context: Context) {

    private val TAG = "TextToSpeechService"

    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var onSpeechComplete: (() -> Unit)? = null

    /**
     * Initialize TTS engine
     */
    fun initialize(onReady: () -> Unit = {}, onError: ((String) -> Unit)? = null) {
        Log.d(TAG, "🔧 Initializing Text-to-Speech...")

        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Set language to US English
                val result = tts?.setLanguage(Locale.US)

                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "❌ Language not supported")
                    onError?.invoke("Language not supported")
                } else {
                    // Configure TTS
                    tts?.setPitch(1.0f)
                    tts?.setSpeechRate(0.9f) // Slightly slower for clarity

                    // Set up utterance listener
                    setupUtteranceListener()

                    isInitialized = true
                    Log.d(TAG, "✅ Text-to-Speech initialized successfully")
                    onReady()
                }
            } else {
                Log.e(TAG, "❌ Text-to-Speech initialization failed")
                onError?.invoke("TTS initialization failed")
            }
        }
    }

    /**
     * Setup utterance progress listener
     */
    private fun setupUtteranceListener() {
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "🗣️ Started speaking: $utteranceId")
            }

            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "✅ Finished speaking: $utteranceId")
                onSpeechComplete?.invoke()
            }

            override fun onError(utteranceId: String?) {
                Log.e(TAG, "❌ Speech error: $utteranceId")
            }
        })
    }

    /**
     * Speak text
     * @param text The text to speak
     * @param utteranceId Unique ID for this utterance
     * @param onComplete Callback when speech completes
     */
    fun speak(
        text: String,
        utteranceId: String = UUID.randomUUID().toString(),
        onComplete: (() -> Unit)? = null
    ) {
        if (!isInitialized) {
            Log.e(TAG, "❌ TTS not initialized")
            onComplete?.invoke()
            return
        }

        Log.d(TAG, "🗣️ Speaking: '$text'")

        onSpeechComplete = onComplete

        val params = Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
        }

        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)
    }

    /**
     * Stop speaking
     */
    fun stop() {
        tts?.stop()
        Log.d(TAG, "🛑 Speech stopped")
    }

    /**
     * Check if TTS is speaking
     */
    fun isSpeaking(): Boolean {
        return tts?.isSpeaking ?: false
    }

    /**
     * Release TTS resources
     */
    fun shutdown() {
        Log.d(TAG, "🔌 Shutting down Text-to-Speech...")
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
        Log.d(TAG, "✅ Text-to-Speech shut down")
    }
}
package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import java.util.*

data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val confidence: Float,
    val department: String,
    val status: String,
    val timestamp: String
)

// ── Theme Colors ──
private val SeraLightColors = lightColorScheme(
    primary = Color(0xFF1565C0),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFE3F2FD),
    onPrimaryContainer = Color(0xFF0D47A1),
    secondary = Color(0xFF00897B),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE0F2F1),
    onSecondaryContainer = Color(0xFF004D40),
    tertiary = Color(0xFFFF8F00),
    onTertiary = Color.White,
    error = Color(0xFFD32F2F),
    background = Color(0xFFF5F5F5),
    surface = Color.White,
    surfaceVariant = Color(0xFFF0F0F0),
    onBackground = Color(0xFF212121),
    onSurface = Color(0xFF212121),
    onSurfaceVariant = Color(0xFF757575)
)

private val SeraDarkColors = darkColorScheme(
    primary = Color(0xFF64B5F6),
    onPrimary = Color(0xFF0D47A1),
    primaryContainer = Color(0xFF1A237E),
    onPrimaryContainer = Color(0xFFBBDEFB),
    secondary = Color(0xFF80CBC4),
    onSecondary = Color(0xFF004D40),
    secondaryContainer = Color(0xFF1B3B38),
    onSecondaryContainer = Color(0xFFB2DFDB),
    tertiary = Color(0xFFFFB74D),
    onTertiary = Color(0xFF4E342E),
    error = Color(0xFFEF9A9A),
    background = Color(0xFF121212),
    surface = Color(0xFF1E1E1E),
    surfaceVariant = Color(0xFF2C2C2C),
    onBackground = Color(0xFFE0E0E0),
    onSurface = Color(0xFFE0E0E0),
    onSurfaceVariant = Color(0xFFBDBDBD)
)

// ── Department Colors ──
fun getDepartmentColor(department: String): Color = when (department) {
    "Housekeeping" -> Color(0xFF4CAF50)
    "Room Service" -> Color(0xFFFF9800)
    "Maintenance" -> Color(0xFF2196F3)
    "Front Desk" -> Color(0xFF9C27B0)
    "Concierge" -> Color(0xFF00BCD4)
    else -> Color(0xFF757575)
}

fun getStatusBorderColor(status: String): Color = when (status) {
    "pending" -> Color(0xFFFFC107)
    "in_progress" -> Color(0xFF2196F3)
    "completed" -> Color(0xFF4CAF50)
    else -> Color.Gray
}

// ── Relative Time Helper ──
fun getRelativeTime(timestamp: String): String {
    try {
        if (timestamp.contains("T") || timestamp.contains("-")) {
            val parts = timestamp.replace("T", " ").split(" ")
            if (parts.size >= 2) {
                val dateParts = parts[0].split("-")
                val timeParts = parts[1].split(":")
                if (dateParts.size == 3 && timeParts.size >= 2) {
                    val cal = Calendar.getInstance()
                    cal.set(dateParts[0].toInt(), dateParts[1].toInt() - 1, dateParts[2].toInt(),
                        timeParts[0].toInt(), timeParts[1].toInt(),
                        if (timeParts.size > 2) timeParts[2].split(".")[0].toInt() else 0)
                    val diffMs = System.currentTimeMillis() - cal.timeInMillis
                    val diffMin = diffMs / 60000
                    return when {
                        diffMin < 1 -> "Just now"
                        diffMin < 60 -> "${diffMin}m ago"
                        diffMin < 1440 -> "${diffMin / 60}h ago"
                        else -> "${diffMin / 1440}d ago"
                    }
                }
            }
        }
    } catch (_: Exception) {}
    return timestamp
}

class MainActivity : ComponentActivity() {
    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private val _apiService = mutableStateOf<ApiService?>(null)
    private var apiService: ApiService
        get() = _apiService.value!!
        set(value) { _apiService.value = value }
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val TAG = "MainActivity"
    private val ROOM_NUMBER = "101"
    private val _requestHistory = mutableStateListOf<RequestItem>()

    // Network & server config state
    private val _wifiSsid = mutableStateOf<String?>(null)
    private val _deviceIp = mutableStateOf<String?>(null)
    private val _activeProfileIndex = mutableIntStateOf(0)
    private val _profiles = mutableStateListOf<NetworkProfile>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeServices()
        } else {
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_LONG).show()
        }
    }

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            refreshNetworkInfo()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load server config from SharedPreferences
        ServerConfig.load(this)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        _activeProfileIndex.intValue = ServerConfig.activeProfileIndex

        voskService = VoskService(this)
        audioRecorder = AudioRecorder(this)
        nluService = NLUService(this)
        apiService = ApiService(ServerConfig.baseUrl)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "TTS: Language not supported")
                } else {
                    tts.setSpeechRate(0.9f)
                    tts.setPitch(1.0f)
                    ttsReady = true
                    Log.d(TAG, "TTS Initialized")
                }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            initializeServices()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            refreshNetworkInfo()
        } else {
            locationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        connectWebSocket()

        setContent {
            VoiceAssistantScreen(
                roomNumber = ROOM_NUMBER,
                voskService = voskService,
                audioRecorder = audioRecorder,
                nluService = nluService,
                apiService = _apiService.value ?: return@setContent,
                lifecycleScope = lifecycleScope,
                requestHistory = _requestHistory,
                wifiSsid = _wifiSsid,
                deviceIp = _deviceIp,
                profiles = _profiles,
                activeProfileIndex = _activeProfileIndex,
                onSpeakAndWait = { message -> speakAndWait(message) },
                onSpeakResponse = { message -> speakFire(message) },
                onAddRequest = { request -> _requestHistory.add(0, request) },
                onCloseApp = { finish() },
                onRefreshRequests = { refreshRequests() },
                onProfileSwitch = { index -> switchProfile(index) },
                onProfileUpdate = { index, profile -> updateProfile(index, profile) },
                onProfileAdd = { profile -> addProfile(profile) },
                onProfileRemove = { index -> removeProfile(index) }
            )
        }

        refreshRequests()
    }

    private fun refreshRequests() {
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                }
            },
            onError = { error ->
                runOnUiThread {
                    Toast.makeText(this, "Sync failed", Toast.LENGTH_SHORT).show()
                    speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
                }
            }
        )
    }

    private fun initializeServices() {
        Log.d(TAG, "Initializing services...")
        lifecycleScope.launch {
            voskService.initialize()
        }
        nluService.initialize()
    }

    private fun refreshNetworkInfo() {
        _wifiSsid.value = NetworkUtils.getWifiSsid(this)
        _deviceIp.value = NetworkUtils.getDeviceIp(this)
        Log.d(TAG, "Network: SSID=${_wifiSsid.value}, IP=${_deviceIp.value}")
    }

    private fun switchProfile(index: Int) {
        ServerConfig.switchProfile(this, index)
        _activeProfileIndex.intValue = index
        reconnectToServer()
        Toast.makeText(this, "Switched to: ${ServerConfig.activeProfile.name}", Toast.LENGTH_SHORT).show()
    }

    private fun updateProfile(index: Int, profile: NetworkProfile) {
        ServerConfig.updateProfile(this, index, profile)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        if (index == _activeProfileIndex.intValue) {
            reconnectToServer()
        }
    }

    private fun addProfile(profile: NetworkProfile) {
        ServerConfig.addProfile(this, profile)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
    }

    private fun removeProfile(index: Int) {
        ServerConfig.removeProfile(this, index)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        _activeProfileIndex.intValue = ServerConfig.activeProfileIndex
        reconnectToServer()
    }

    private fun speakFire(message: String) {
        if (ttsReady) {
            val params = Bundle()
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "fire")
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "fire")
        }
    }

    private suspend fun speakAndWait(message: String) {
        if (!ttsReady) return
        val deferred = CompletableDeferred<Unit>()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { deferred.complete(Unit) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { deferred.complete(Unit) }
        })
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "greeting")
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "greeting")
        deferred.await()
    }

    private fun reconnectToServer() {
        apiService = ApiService(ServerConfig.baseUrl)
        if (::webSocketService.isInitialized) {
            webSocketService.disconnect()
        }
        connectWebSocket()
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                    speakFire("Successfully connected to ${ServerConfig.activeProfile.name}. System is ready.")
                }
            },
            onError = { error ->
                runOnUiThread {
                    speakFire("Failed to connect to ${ServerConfig.activeProfile.name}. Please check the network connection or contact the help desk using the land line phone.")
                }
            }
        )
        Log.d(TAG, "Server config updated: ${ServerConfig.baseUrl}")
    }

    private fun connectWebSocket() {
        try {
            webSocketService = WebSocketService(ROOM_NUMBER, ServerConfig.wsUrl(ROOM_NUMBER))
            webSocketService.connect(
                onMessage = { message ->
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
                    }
                },
                onStatusChange = { requestId, status ->
                    Log.d(TAG, "WebSocket status change: request $requestId -> $status")
                    runOnUiThread {
                        val index = _requestHistory.indexOfFirst { it.id == requestId }
                        if (index != -1) {
                            val oldRequest = _requestHistory[index]
                            _requestHistory[index] = oldRequest.copy(status = status)

                            if (ttsReady) {
                                val statusMessage = when (status) {
                                    "in_progress" -> "Your request No.$requestId is now being processed."
                                    "completed" -> "Your request No.$requestId is completed. Thank you for your patience!"
                                    else -> "Your request No.$requestId is now $status."
                                }
                                val params = Bundle()
                                params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "StatusUpdate_$requestId")
                                tts.speak(statusMessage, TextToSpeech.QUEUE_FLUSH, params, "StatusUpdate_$requestId")
                            }
                        } else {
                            refreshRequests()
                        }
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "WebSocket error", e)
            runOnUiThread {
                speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        voskService.release()
        nluService.close()
        webSocketService.disconnect()
    }
}

// ── Animated Mic Button ──
@Composable
fun AnimatedMicButton(
    isRecording: Boolean,
    isProcessing: Boolean,
    audioLevel: Float,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "mic")

    // Pulse animation while recording
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(600, easing = EaseInOut),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    // Spinner rotation while processing
    val spinAngle by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing)
        ),
        label = "spin"
    )

    // Glow alpha
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.2f,
        targetValue = 0.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(800, easing = EaseInOut),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )

    val currentScale = when {
        isRecording -> pulseScale
        else -> 1f
    }

    val buttonColor = when {
        isRecording -> Color(0xFFD32F2F)
        isProcessing -> Color(0xFFFF8F00)
        else -> Color(0xFF1565C0)
    }

    val glowColor = when {
        isRecording -> Color(0xFFD32F2F)
        isProcessing -> Color(0xFFFF8F00)
        else -> Color(0xFF1565C0)
    }

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier.size(120.dp)
    ) {
        // Audio level rings (visible during recording)
        if (isRecording) {
            val normalizedLevel = (audioLevel * 8f).coerceIn(0f, 1f)
            Canvas(modifier = Modifier.size(120.dp)) {
                val ringRadius = size.minDimension / 2f
                drawCircle(
                    color = glowColor.copy(alpha = glowAlpha * normalizedLevel),
                    radius = ringRadius * (1f + normalizedLevel * 0.3f),
                    style = Stroke(width = 3.dp.toPx())
                )
                drawCircle(
                    color = glowColor.copy(alpha = glowAlpha * normalizedLevel * 0.5f),
                    radius = ringRadius * (1f + normalizedLevel * 0.5f),
                    style = Stroke(width = 2.dp.toPx())
                )
            }
        }

        // Processing spinner
        if (isProcessing) {
            Canvas(modifier = Modifier.size(100.dp)) {
                drawArc(
                    color = glowColor.copy(alpha = 0.7f),
                    startAngle = spinAngle,
                    sweepAngle = 120f,
                    useCenter = false,
                    topLeft = Offset.Zero,
                    size = Size(size.width, size.height),
                    style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
                )
            }
        }

        // Main button
        Surface(
            onClick = { if (!isRecording && !isProcessing) onClick() },
            modifier = Modifier
                .size(88.dp)
                .scale(currentScale)
                .shadow(
                    elevation = if (isRecording) 12.dp else 6.dp,
                    shape = CircleShape,
                    ambientColor = glowColor.copy(alpha = 0.3f),
                    spotColor = glowColor.copy(alpha = 0.3f)
                ),
            shape = CircleShape,
            color = buttonColor
        ) {
            Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                // Mic icon drawn with Canvas
                Canvas(modifier = Modifier.size(36.dp)) {
                    val w = size.width
                    val h = size.height
                    val iconColor = Color.White
                    val strokeW = 2.5.dp.toPx()

                    if (isRecording) {
                        // Recording: filled circle (stop indicator)
                        drawRoundRect(
                            color = iconColor,
                            topLeft = Offset(w * 0.25f, h * 0.25f),
                            size = Size(w * 0.5f, h * 0.5f),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx())
                        )
                    } else if (isProcessing) {
                        // Processing: gear-like dots
                        for (i in 0 until 3) {
                            drawCircle(
                                color = iconColor,
                                radius = 3.dp.toPx(),
                                center = Offset(w * (0.25f + i * 0.25f), h * 0.5f)
                            )
                        }
                    } else {
                        // Mic icon
                        // Mic body
                        drawRoundRect(
                            color = iconColor,
                            topLeft = Offset(w * 0.35f, h * 0.1f),
                            size = Size(w * 0.3f, h * 0.45f),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(w * 0.15f)
                        )
                        // Mic arc
                        drawArc(
                            color = iconColor,
                            startAngle = 0f,
                            sweepAngle = 180f,
                            useCenter = false,
                            topLeft = Offset(w * 0.22f, h * 0.2f),
                            size = Size(w * 0.56f, h * 0.55f),
                            style = Stroke(width = strokeW, cap = StrokeCap.Round)
                        )
                        // Stem
                        drawLine(
                            color = iconColor,
                            start = Offset(w * 0.5f, h * 0.75f),
                            end = Offset(w * 0.5f, h * 0.88f),
                            strokeWidth = strokeW,
                            cap = StrokeCap.Round
                        )
                        // Base
                        drawLine(
                            color = iconColor,
                            start = Offset(w * 0.35f, h * 0.88f),
                            end = Offset(w * 0.65f, h * 0.88f),
                            strokeWidth = strokeW,
                            cap = StrokeCap.Round
                        )
                    }
                }
            }
        }
    }
}

// ── Audio Level Indicator ──
@Composable
fun AudioLevelIndicator(audioLevel: Float, isRecording: Boolean, modifier: Modifier = Modifier) {
    if (!isRecording) return
    val normalizedLevel = (audioLevel * 10f).coerceIn(0f, 1f)

    Row(
        modifier = modifier.height(24.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        val barCount = 20
        for (i in 0 until barCount) {
            val threshold = i.toFloat() / barCount
            val isActive = normalizedLevel > threshold
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight(if (isActive) 0.3f + (normalizedLevel - threshold).coerceIn(0f, 0.7f) else 0.2f)
                    .clip(RoundedCornerShape(2.dp))
                    .background(
                        if (isActive) {
                            when {
                                i < barCount * 0.6 -> Color(0xFF4CAF50)
                                i < barCount * 0.8 -> Color(0xFFFFC107)
                                else -> Color(0xFFD32F2F)
                            }
                        } else {
                            Color(0xFF424242)
                        }
                    )
            )
        }
    }
}

@Composable
fun VoiceAssistantScreen(
    roomNumber: String,
    voskService: VoskService,
    audioRecorder: AudioRecorder,
    nluService: NLUService,
    apiService: ApiService,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    requestHistory: List<RequestItem>,
    wifiSsid: State<String?>,
    deviceIp: State<String?>,
    profiles: List<NetworkProfile>,
    activeProfileIndex: MutableIntState,
    onSpeakAndWait: suspend (String) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onCloseApp: () -> Unit,
    onRefreshRequests: () -> Unit,
    onProfileSwitch: (Int) -> Unit,
    onProfileUpdate: (Int, NetworkProfile) -> Unit,
    onProfileAdd: (NetworkProfile) -> Unit,
    onProfileRemove: (Int) -> Unit
) {
    val context = LocalContext.current
    var isRecording by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Tap microphone to start") }
    var lastTranscription by remember { mutableStateOf("") }
    var lastIntent by remember { mutableStateOf("") }
    var lastConfidence by remember { mutableStateOf(0f) }
    var lastDepartment by remember { mutableStateOf("") }
    var showServerDialog by remember { mutableStateOf(false) }
    var settingsExpanded by remember { mutableStateOf(false) }
    var audioLevel by remember { mutableFloatStateOf(0f) }

    // Sort requests: in_progress first, then pending, then completed
    val sortedRequests = remember(requestHistory) {
        requestHistory.sortedWith(compareBy {
            when (it.status) {
                "in_progress" -> 0
                "pending" -> 1
                "completed" -> 2
                else -> 3
            }
        })
    }

    val isDark = isSystemInDarkTheme()
    val colorScheme = if (isDark) SeraDarkColors else SeraLightColors

    MaterialTheme(colorScheme = colorScheme) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize()) {
                Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {

                    // ── Collapsible Settings Panel ──
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(modifier = Modifier.fillMaxWidth()) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column {
                                    Text(
                                        text = "Sera - Voice Assistant",
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onPrimaryContainer
                                    )
                                    Text(
                                        text = "Room $roomNumber",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                                    )
                                }
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    IconButton(onClick = { settingsExpanded = !settingsExpanded }, modifier = Modifier.size(32.dp)) {
                                        Icon(
                                            if (settingsExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                            contentDescription = "Toggle Settings",
                                            tint = MaterialTheme.colorScheme.onPrimaryContainer
                                        )
                                    }
                                    IconButton(onClick = onCloseApp, modifier = Modifier.size(32.dp)) {
                                        Icon(Icons.Default.Close, contentDescription = "Close App", tint = MaterialTheme.colorScheme.error)
                                    }
                                }
                            }

                            AnimatedVisibility(visible = settingsExpanded) {
                                Card(
                                    modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = if (isDark) Color(0xFF1B3B2F) else Color(0xFFE8F5E9)
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Column(modifier = Modifier.fillMaxWidth().padding(10.dp)) {
                                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                            Text(
                                                text = "WiFi: ${wifiSsid.value ?: "Not connected"}",
                                                style = MaterialTheme.typography.bodySmall,
                                                fontWeight = FontWeight.Medium
                                            )
                                            Text(
                                                text = "IP: ${deviceIp.value ?: "N/A"}",
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            profiles.forEachIndexed { index, profile ->
                                                val isActive = index == activeProfileIndex.intValue
                                                FilterChip(
                                                    selected = isActive,
                                                    onClick = { if (!isActive) onProfileSwitch(index) },
                                                    label = { Text(profile.name, fontSize = 11.sp) },
                                                    modifier = Modifier.weight(1f)
                                                )
                                            }
                                            IconButton(onClick = { showServerDialog = true }, modifier = Modifier.size(28.dp)) {
                                                Icon(Icons.Default.Edit, contentDescription = "Edit Profiles", modifier = Modifier.size(16.dp))
                                            }
                                        }
                                        val active = profiles.getOrNull(activeProfileIndex.intValue)
                                        if (active != null) {
                                            Text(
                                                text = "Server: ${active.serverIp}:${active.serverPort}",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = if (isDark) Color(0xFF81C784) else Color(0xFF2E7D32)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (showServerDialog) {
                        ProfileEditDialog(
                            profiles = profiles,
                            activeIndex = activeProfileIndex.intValue,
                            onDismiss = { showServerDialog = false },
                            onUpdateProfile = { index, profile -> onProfileUpdate(index, profile) },
                            onAddProfile = { profile -> onProfileAdd(profile) },
                            onRemoveProfile = { index ->
                                onProfileRemove(index)
                                if (profiles.size <= 1) showServerDialog = false
                            }
                        )
                    }

                    Spacer(modifier = Modifier.height(6.dp))

                    // ── Mic Section (1/3) ──
                    Card(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(
                            modifier = Modifier.fillMaxSize().padding(12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            AnimatedMicButton(
                                isRecording = isRecording,
                                isProcessing = isProcessing,
                                audioLevel = audioLevel,
                                onClick = {
                                    if (!isRecording && !isProcessing) {
                                        lifecycleScope.launch {
                                            statusMessage = "Hello! I'm Sera. How can I help you?"
                                            onSpeakAndWait("Hello! I'm Sera. How can I help you?")
                                            processVoiceRequest(
                                                audioRecorder, voskService, nluService, apiService, roomNumber, lifecycleScope,
                                                { isRecording = true }, { isRecording = false },
                                                { isProcessing = true }, { isProcessing = false },
                                                { statusMessage = it }, { lastTranscription = it },
                                                { i, c -> lastIntent = i; lastConfidence = c },
                                                { level -> audioLevel = level },
                                                onSpeakResponse,
                                                { request -> lastDepartment = request.department; onAddRequest(request) },
                                                { statusMessage = "Tap microphone to start"; audioLevel = 0f }
                                            )
                                        }
                                    }
                                }
                            )

                            Spacer(modifier = Modifier.height(4.dp))

                            // Audio level bar
                            AudioLevelIndicator(
                                audioLevel = audioLevel,
                                isRecording = isRecording,
                                modifier = Modifier.fillMaxWidth(0.7f)
                            )

                            Text(
                                text = when {
                                    isRecording -> "Listening..."
                                    isProcessing -> "Processing..."
                                    else -> "Tap to speak"
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                            )

                            if (lastTranscription.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(4.dp))
                                HorizontalDivider(color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.2f))
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "\"$lastTranscription\"",
                                    fontWeight = FontWeight.Medium,
                                    fontSize = 13.sp,
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis,
                                    color = MaterialTheme.colorScheme.onSecondaryContainer
                                )
                                if (lastIntent.isNotEmpty()) {
                                    Row(
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = "Intent: $lastIntent (${(lastConfidence * 100).toInt()}%)",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                                        )
                                    }
                                }
                                if (lastDepartment.isNotEmpty()) {
                                    DepartmentBadge(department = lastDepartment)
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(6.dp))

                    // ── Recent Requests Header ──
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Recent Requests",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        IconButton(onClick = onRefreshRequests, modifier = Modifier.size(28.dp)) {
                            Icon(
                                Icons.Default.Refresh,
                                contentDescription = "Refresh",
                                modifier = Modifier.size(18.dp),
                                tint = MaterialTheme.colorScheme.onBackground
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // ── Requests List (2/3) ──
                    Box(modifier = Modifier.fillMaxWidth().weight(2f)) {
                        if (sortedRequests.isEmpty()) {
                            // Better empty state
                            Card(
                                modifier = Modifier.fillMaxSize(),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        // Mic icon as empty state illustration
                                        Canvas(modifier = Modifier.size(56.dp)) {
                                            val w = size.width
                                            val h = size.height
                                            val col = Color(0xFFBDBDBD)
                                            val sw = 3.dp.toPx()
                                            drawRoundRect(col, Offset(w*0.35f,h*0.1f), Size(w*0.3f,h*0.45f), androidx.compose.ui.geometry.CornerRadius(w*0.15f))
                                            drawArc(col, 0f, 180f, false, Offset(w*0.2f,h*0.15f), Size(w*0.6f,h*0.55f), style = Stroke(sw, cap = StrokeCap.Round))
                                            drawLine(col, Offset(w*0.5f,h*0.7f), Offset(w*0.5f,h*0.85f), sw, StrokeCap.Round)
                                            drawLine(col, Offset(w*0.32f,h*0.85f), Offset(w*0.68f,h*0.85f), sw, StrokeCap.Round)
                                        }
                                        Spacer(modifier = Modifier.height(12.dp))
                                        Text(
                                            text = "No requests yet",
                                            style = MaterialTheme.typography.titleSmall,
                                            fontWeight = FontWeight.Medium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "Tap the microphone button above\nto make your first request",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                                            lineHeight = 18.sp
                                        )
                                    }
                                }
                            }
                        } else {
                            LazyColumn(
                                modifier = Modifier.fillMaxSize(),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                items(sortedRequests) { request ->
                                    RequestCard(request = request)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

suspend fun processVoiceRequest(
    audioRecorder: AudioRecorder,
    voskService: VoskService,
    nluService: NLUService,
    apiService: ApiService,
    roomNumber: String,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    onRecordingStart: () -> Unit,
    onRecordingStop: () -> Unit,
    onProcessingStart: () -> Unit,
    onProcessingStop: () -> Unit,
    onStatusUpdate: (String) -> Unit,
    onTranscriptionUpdate: (String) -> Unit,
    onIntentUpdate: (String, Float) -> Unit,
    onAudioLevel: (Float) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onComplete: () -> Unit
) {
    if (ContextCompat.checkSelfPermission(audioRecorder.getContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        onStatusUpdate("Permission denied")
        onComplete()
        return
    }

    try {
        onRecordingStart(); onStatusUpdate("Listening... (speak now)")

        val audioData = audioRecorder.recordWithVAD(
            silenceTimeoutMs = 1500L,
            maxDurationMs = 10000L,
            onStateChange = { state -> onStatusUpdate(state) },
            onAudioLevel = { level -> onAudioLevel(level) }
        )

        onAudioLevel(0f)
        onRecordingStop(); onProcessingStart(); onStatusUpdate("Processing...")

        lifecycleScope.launch {
            try {
                onStatusUpdate("Transcribing...")
                val transcription = voskService.transcribeAudio(audioData)
                onTranscriptionUpdate(transcription)

                if (transcription.isEmpty()) {
                    onStatusUpdate("No speech detected"); onProcessingStop(); onComplete(); return@launch
                }

                onStatusUpdate("Understanding...")
                val cleanedTranscription = cleanTranscription(transcription)
                val intentResult = nluService.classifyIntent(cleanedTranscription)
                onIntentUpdate(intentResult.name, intentResult.confidence)

                // Reject low-confidence or irrelevant requests
                val MIN_CONFIDENCE = 0.60f
                if (intentResult.confidence < MIN_CONFIDENCE) {
                    onStatusUpdate("Request not understood")
                    onSpeakResponse("I'm sorry, I couldn't understand your request. Could you please try again with a specific hotel service request, such as requesting towels, room cleaning, or room service?")
                    onProcessingStop(); onComplete(); return@launch
                }

                onStatusUpdate("Submitting...")
                apiService.submitRequest(
                    roomNumber = roomNumber, requestText = transcription, intent = intentResult.name,
                    onSuccess = { response ->
                        onAddRequest(RequestItem(response.requestId, transcription, intentResult.name, intentResult.confidence, "Routing...", "pending", getCurrentTime()))
                        val speechText = "Your request No.${response.requestId} has been received."
                        onSpeakResponse(speechText)
                        onProcessingStop(); onComplete()
                    },
                    onError = {
                        onStatusUpdate("Submission failed")
                        onSpeakResponse("Sorry, I could not send your request due to a network issue. Please contact the help desk using the land line phone.")
                        onProcessingStop(); onComplete()
                    }
                )
            } catch (e: Exception) { onStatusUpdate("Error occurred"); onProcessingStop(); onComplete() }
        }
    } catch (e: Exception) { onRecordingStop(); onProcessingStop(); onComplete() }
}

// ── Department Badge ──
@Composable
fun DepartmentBadge(department: String) {
    val color = getDepartmentColor(department)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(top = 2.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = department,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            color = color
        )
    }
}

// ── Request Card with colored left border ──
@Composable
fun RequestCard(request: RequestItem) {
    val borderColor = getStatusBorderColor(request.status)
    val deptColor = getDepartmentColor(request.department)

    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(modifier = Modifier.fillMaxWidth()) {
            // Status color bar on the left
            Box(
                modifier = Modifier
                    .width(4.dp)
                    .fillMaxHeight()
                    .background(borderColor)
            )
            Column(modifier = Modifier.fillMaxWidth().padding(10.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = "#${request.id}", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    StatusBadge(status = request.status)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = request.requestText,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Department badge
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(6.dp)
                                .clip(CircleShape)
                                .background(deptColor)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = request.department,
                            fontSize = 11.sp,
                            color = deptColor,
                            fontWeight = FontWeight.Medium
                        )
                    }
                    Text(
                        text = getRelativeTime(request.timestamp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
fun StatusBadge(status: String) {
    val (backgroundColor, textColor, text) = when (status) {
        "pending" -> Triple(Color(0xFFFFF3CD), Color(0xFF856404), "Pending")
        "in_progress" -> Triple(Color(0xFFD1ECF1), Color(0xFF0C5460), "In Progress")
        "completed" -> Triple(Color(0xFFD4EDDA), Color(0xFF155724), "Completed")
        else -> Triple(Color.Gray, Color.White, status)
    }
    Box(modifier = Modifier.background(backgroundColor, RoundedCornerShape(12.dp)).padding(horizontal = 8.dp, vertical = 2.dp)) {
        Text(text = text, color = textColor, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ProfileEditDialog(
    profiles: List<NetworkProfile>,
    activeIndex: Int,
    onDismiss: () -> Unit,
    onUpdateProfile: (Int, NetworkProfile) -> Unit,
    onAddProfile: (NetworkProfile) -> Unit,
    onRemoveProfile: (Int) -> Unit
) {
    var editingIndex by remember { mutableStateOf<Int?>(null) }
    var editName by remember { mutableStateOf("") }
    var editIp by remember { mutableStateOf("") }
    var editPort by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Network Profiles") },
        text = {
            Column {
                profiles.forEachIndexed { index, profile ->
                    if (editingIndex == index) {
                        OutlinedTextField(
                            value = editName,
                            onValueChange = { editName = it },
                            label = { Text("Name") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        OutlinedTextField(
                            value = editIp,
                            onValueChange = { editIp = it },
                            label = { Text("Server IP") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        OutlinedTextField(
                            value = editPort,
                            onValueChange = { editPort = it },
                            label = { Text("Port") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = {
                                onUpdateProfile(index, NetworkProfile(editName.trim(), editIp.trim(), editPort.toIntOrNull() ?: 8000))
                                editingIndex = null
                            }) { Text("Save") }
                            TextButton(onClick = { editingIndex = null }) { Text("Cancel") }
                        }
                    } else {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = profile.name + if (index == activeIndex) " (Active)" else "",
                                    fontWeight = if (index == activeIndex) FontWeight.Bold else FontWeight.Normal,
                                    fontSize = 14.sp
                                )
                                Text(
                                    text = "${profile.serverIp}:${profile.serverPort}",
                                    fontSize = 12.sp,
                                    color = Color.Gray
                                )
                            }
                            IconButton(onClick = {
                                editingIndex = index
                                editName = profile.name
                                editIp = profile.serverIp
                                editPort = profile.serverPort.toString()
                            }, modifier = Modifier.size(32.dp)) {
                                Icon(Icons.Default.Edit, contentDescription = "Edit", modifier = Modifier.size(16.dp))
                            }
                            if (profiles.size > 1) {
                                IconButton(onClick = { onRemoveProfile(index) }, modifier = Modifier.size(32.dp)) {
                                    Icon(Icons.Default.Close, contentDescription = "Remove", modifier = Modifier.size(16.dp), tint = Color.Red)
                                }
                            }
                        }
                    }
                    if (index < profiles.lastIndex) HorizontalDivider()
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onAddProfile(NetworkProfile("New Network", "192.168.1.100", 8000))
            }) { Text("Add Profile") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        }
    )
}

fun cleanTranscription(text: String): String {
    val prefixPatterns = listOf(
        "hi sera", "hey sera", "hello sera", "sera",
        "hi there", "hey there", "hello there",
        "hi", "hey", "hello",
        "excuse me", "please", "can you", "could you", "i need you to"
    )
    var cleaned = text.lowercase().trim()
    for (pattern in prefixPatterns) {
        if (cleaned.startsWith(pattern)) {
            cleaned = cleaned.removePrefix(pattern).trim()
            cleaned = cleaned.removePrefix(",").removePrefix(".").trim()
            break
        }
    }
    return cleaned.ifEmpty { text.trim() }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    return String.format(Locale.getDefault(), "%02d:%02d", calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE))
}

fun AudioRecorder.getContext(): android.content.Context {
    val field = this.javaClass.getDeclaredField("context")
    field.isAccessible = true
    return field.get(this) as android.content.Context
}

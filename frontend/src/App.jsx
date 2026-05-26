import { useState } from 'react'
import { Upload, ShieldAlert, CheckCircle2, Activity, ShieldCheck, X } from 'lucide-react'
import axios from 'axios'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleClear = (e) => {
    e.preventDefault()
    setFile(null)
    setResult(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Assuming backend will be running on port 8000
      const response = await axios.post('http://localhost:8000/api/detect/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(response.data)
    } catch (error) {
      console.error('Error uploading file:', error)
      alert("Failed to connect to the backend API. Make sure it's running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-200 font-sans p-6 selection:bg-cyan-500/30">
      <div className="max-w-5xl mx-auto">
        <header className="mb-12 text-center py-10">
          <div className="inline-flex items-center justify-center p-3 bg-cyan-500/10 rounded-2xl mb-4 shadow-[0_0_30px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/30">
             <ShieldCheck className="w-10 h-10 text-cyan-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500 mb-4">
            DeepGuard
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto font-light">
            Multi-Modal Deepfake Detection & Media Verification System
          </p>
        </header>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
              <Upload className="w-5 h-5 text-cyan-400" />
              Upload Media
            </h2>
            
            <div className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 transition-colors bg-slate-900/50 rounded-2xl p-10 text-center relative z-10 flex flex-col items-center justify-center min-h-[240px]">
              {!file && (
                <input 
                  type="file" 
                  onChange={handleFileChange} 
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept="video/*,audio/*,image/*"
                />
              )}
              
              {!file ? (
                <div className="flex flex-col items-center gap-3 pointer-events-none">
                  <div className="p-4 bg-slate-800 rounded-full text-slate-400">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm text-slate-300 font-medium">
                    Drag & drop or click to browse
                  </p>
                  <p className="text-xs text-slate-500">
                    Supports MP4, AVI, WAV, MP3, JPG, PNG
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 relative z-20">
                  <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-cyan-400">
                    <CheckCircle2 className="w-8 h-8" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-slate-200 font-medium break-all max-w-[250px] line-clamp-2">
                      {file.name}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <button 
                    onClick={handleClear}
                    className="mt-2 flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-red-500/20 text-slate-300 hover:text-red-400 rounded-lg text-sm transition-colors border border-slate-700 hover:border-red-500/50"
                  >
                    <X className="w-4 h-4" />
                    Remove Media
                  </button>
                </div>
              )}
            </div>

            <button 
              onClick={handleUpload}
              disabled={!file || loading}
              className={`w-full mt-6 py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all relative z-10 ${
                !file || loading 
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:-translate-y-0.5'
              }`}
            >
              {loading ? (
                <><Activity className="w-5 h-5 animate-spin" /> Analyzing...</>
              ) : (
                'Run Analysis'
              )}
            </button>
          </div>

          {/* Results Section */}
          <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 shadow-2xl relative">
            <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              Analysis Results
            </h2>

            {!result && !loading && (
              <div className="flex flex-col items-center justify-center h-48 text-slate-500">
                <Activity className="w-8 h-8 mb-3 opacity-20" />
                <p>Upload media to view deepfake analysis</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center h-48 text-cyan-400 space-y-4">
                <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin"></div>
                <p className="animate-pulse font-medium">Running multi-modal inference...</p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className={`p-6 rounded-2xl border ${result.prediction === 'FAKE' ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'} flex items-start gap-4`}>
                  {result.prediction === 'FAKE' ? <ShieldAlert className="w-8 h-8 shrink-0" /> : <CheckCircle2 className="w-8 h-8 shrink-0" />}
                  <div>
                    <h3 className="text-xl font-bold mb-1">
                      {result.prediction === 'FAKE' ? 'Deepfake Detected' : 'Authentic Media'}
                    </h3>
                    <p className="text-sm opacity-80">
                      Overall Confidence: {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">Modal Breakdown</h4>
                  
                  <div className="space-y-3">
                    {/* Visual */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">Visual Artifacts</span>
                        <span className="text-slate-400 font-mono">{(result.details.visual_score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-400" style={{width: `${result.details.visual_score * 100}%`}}></div>
                      </div>
                    </div>

                    {/* Audio */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">Audio Synthesis</span>
                        <span className="text-slate-400 font-mono">{(result.details.audio_score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-purple-500 to-pink-400" style={{width: `${result.details.audio_score * 100}%`}}></div>
                      </div>
                    </div>

                    {/* Temporal */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">Temporal Inconsistency</span>
                        <span className="text-slate-400 font-mono">{(result.details.temporal_score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-orange-500 to-amber-400" style={{width: `${result.details.temporal_score * 100}%`}}></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Extracted Faces Visualization */}
                {result.extracted_faces && result.extracted_faces.length > 0 && (
                  <div className="pt-6 mt-6 border-t border-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
                    <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase mb-4 flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-cyan-400" /> Faces Extracted & Analyzed
                    </h4>
                    <div className="grid grid-cols-5 gap-3">
                      {result.extracted_faces.map((face, idx) => (
                        <div key={idx} className="aspect-square rounded-xl overflow-hidden border border-slate-700/50 bg-slate-800/50 shadow-inner group relative">
                          <img 
                            src={face} 
                            alt={`Analyzed Face ${idx + 1}`} 
                            className="w-full h-full object-cover group-hover:scale-110 group-hover:brightness-110 transition-all duration-500 ease-out" 
                          />
                          <div className="absolute inset-0 ring-1 ring-inset ring-white/10 rounded-xl group-hover:ring-cyan-500/50 transition-colors"></div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

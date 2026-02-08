import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')
  const [path, setPath] = useState('')

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setStatus('Please select a file first')
      return
    }

    setStatus('Uploading...')
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Assuming backend is running on localhost:8000
      const response = await axios.post('http://localhost:8000/api/v1/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setStatus('Upload successful: ' + response.data.message)
      setPath(response.data.file_path)
    } catch (error) {
      console.error(error)
      setStatus('Upload failed: ' + (error.response?.data?.detail || error.message))
    }
  }

  return (
    <div className="container">
      <h1>Reference-Based Super-Resolution</h1>
      <div className="card">
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={handleUpload}>
          Upload Video
        </button>
        <p className="status">{status}</p>
        {path && <p className="path">Saved at: {path}</p>}
      </div>
    </div>
  )
}

export default App

import { useEffect, useMemo, useState, type DragEvent } from 'react'

type SampleImage = {
  dataset: string
  className: string
  imagePath: string
  imageUrl: string
  fileName: string
}

type PredictionItem = {
  className: string
  probability: number
}

type PredictResponse = {
  sourceName: string
  predictedClass: string
  confidence: number
  predictions: PredictionItem[]
  classes: string[]
}

type SelectedSource =
  | { kind: 'upload'; name: string; previewUrl: string }
  | { kind: 'sample'; name: string; previewUrl: string; imagePath: string; dataset: string }
  | null

const classLabels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

function App() {
  const [samples, setSamples] = useState<SampleImage[]>([])
  const [kaggleSamples, setKaggleSamples] = useState<SampleImage[]>([])
  const [selectedSource, setSelectedSource] = useState<SelectedSource>(null)
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingSamples, setIsLoadingSamples] = useState(false)
  const [isLoadingKaggleSamples, setIsLoadingKaggleSamples] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadSamples = async () => {
    setIsLoadingSamples(true)
    try {
      const response = await fetch(`/api/fer-samples?t=${Date.now()}`, { cache: 'no-store' })
      if (!response.ok) {
        throw new Error('Can not load FER dataset')
      }

      const data = (await response.json()) as { samples: SampleImage[] }
      setSamples(data.samples)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Can not load FER dataset')
    } finally {
      setIsLoadingSamples(false)
    }
  }

  const loadKaggleSamples = async () => {
    setIsLoadingKaggleSamples(true)
    try {
      const response = await fetch(`/api/kaggle-samples?t=${Date.now()}`, { cache: 'no-store' })
      if (!response.ok) {
        throw new Error('Can not load Kaggle dataset')
      }

      const data = (await response.json()) as { samples: SampleImage[] }
      setKaggleSamples(data.samples)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Can not load Kaggle dataset')
    } finally {
      setIsLoadingKaggleSamples(false)
    }
  }

  useEffect(() => {
    void loadSamples()
    void loadKaggleSamples()
  }, [])

  useEffect(() => {
    return () => {
      if (selectedSource?.kind === 'upload') {
        URL.revokeObjectURL(selectedSource.previewUrl)
      }
    }
  }, [selectedSource])

  const topPrediction = prediction?.predictions[0]

  const rankedPredictions = useMemo(() => {
    if (!prediction) {
      return classLabels.map((className) => ({ className, probability: 0 }))
    }

    const ordered = [...prediction.predictions].sort((left, right) => right.probability - left.probability)
    const byClass = new Map(ordered.map((item) => [item.className, item.probability]))

    return classLabels
      .map((className) => ({ className, probability: byClass.get(className) ?? 0 }))
      .sort((left, right) => right.probability - left.probability)
  }, [prediction])

  const runPrediction = async (formData: FormData, source: SelectedSource) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Can not predict image')
      }

      const data = (await response.json()) as PredictResponse
      setPrediction(data)
      setSelectedSource(source)
    } catch (predictError) {
      setError(predictError instanceof Error ? predictError.message : 'Can not predict image')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const previewUrl = URL.createObjectURL(file)
    setSelectedSource({ kind: 'upload', name: file.name, previewUrl })

    await runPrediction(formData, { kind: 'upload', name: file.name, previewUrl })
  }

  const handleSampleSelect = async (sample: SampleImage) => {
    const formData = new FormData()
    formData.append('image_path', sample.imagePath)
    formData.append('dataset', sample.dataset)

    await runPrediction(formData, {
      kind: 'sample',
      name: sample.fileName,
      previewUrl: sample.imageUrl,
      imagePath: sample.imagePath,
      dataset: sample.dataset,
    })
  }

  const onDrop = async (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files?.[0]
    if (file) {
      await handleFileUpload(file)
    }
  }

  return (
    <div className="page-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">FER Playground</p>
        </div>
        <div className="status-pill">ONNX + React</div>
      </header>

      <main className="layout">
        <section className="hero-card">
          <div>
            <p className="section-kicker">Upload or sample</p>
            <h3>Predict facial emotion from an uploaded image or a test sample.</h3>
          </div>

          <label
            className={`dropzone ${isLoading ? 'dropzone-busy' : ''}`}
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            <input
              type="file"
              accept="image/*"
              onChange={async (event) => {
                const file = event.target.files?.[0]
                if (file) {
                  await handleFileUpload(file)
                }
              }}
            />
            <span className="dropzone-title">Drop image here or click to upload</span>
            <span className="dropzone-subtitle">Supports JPG, PNG, WEBP and BMP</span>
          </label>
        </section>

        <section className="result-grid">
          <article className="panel image-panel">
            <div className="panel-header">
              <h3>Image</h3>
              <span>{selectedSource ? selectedSource.name : 'No image selected'}</span>
            </div>

            <div className="preview-frame">
              {selectedSource ? (
                <img src={selectedSource.previewUrl} alt={selectedSource.name} />
              ) : (
                <div className="empty-state">
                  <p>Select a file or click a test sample to preview it here.</p>
                </div>
              )}
            </div>

            <div className="meta-row">
              <span>{selectedSource?.kind === 'sample' ? 'Test sample' : selectedSource?.kind === 'upload' ? 'Local upload' : 'Waiting for input'}</span>
            </div>
          </article>

          <article className="panel prediction-panel">
            <div className="panel-header">
              <h3>Confidence Scores</h3>
              <span>{isLoading ? 'Predicting...' : topPrediction ? `${prediction?.predictedClass ?? ''} ${Math.round((prediction?.confidence ?? 0) * 100)}%` : 'Awaiting inference'}</span>
            </div>

            <div className="prediction-list">
              {rankedPredictions.map((item, index) => (
                <div className="prediction-row" key={item.className}>
                  <div className="prediction-rank">{index + 1}</div>
                  <div className="prediction-body">
                    <div className="prediction-topline">
                      <span>{item.className}</span>
                      <strong>{(item.probability * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${Math.max(item.probability * 100, 2)}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className="sample-tray panel">
          <div className="tray-header">
            <div className="tray-title-group">
              <span className="tray-badge">FER</span>
              <h3>Try with FER-2013 test samples</h3>
            </div>
            <div className="tray-actions">
              <span>{isLoadingSamples ? 'Loading 10 samples...' : `${samples.length} images`}</span>
              <button className="refresh-button" type="button" onClick={loadSamples} disabled={isLoadingSamples}>
                Refresh
              </button>
            </div>
          </div>

          <div className="samples-row">
            {samples.map((sample) => (
              <button
                key={sample.imagePath}
                className={`sample-card ${selectedSource?.kind === 'sample' && selectedSource.dataset === sample.dataset && selectedSource.imagePath === sample.imagePath ? 'selected' : ''}`}
                onClick={async () => await handleSampleSelect(sample)}
                type="button"
              >
                <img src={sample.imageUrl} alt={sample.fileName} />
                <div className="sample-caption">
                  <strong>{sample.className}</strong>
                  <span>{sample.fileName}</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="sample-tray panel sample-tray-secondary">
          <div className="tray-header">
            <div className="tray-title-group">
              <span className="tray-badge tray-badge-alt">KAGGLE</span>
              <h3>More Kaggle Samples</h3>
            </div>
            <div className="tray-actions">
              <span>{isLoadingKaggleSamples ? 'Loading 10 samples...' : `${kaggleSamples.length} images`}</span>
              <button className="refresh-button" type="button" onClick={loadKaggleSamples} disabled={isLoadingKaggleSamples}>
                Refresh
              </button>
            </div>
          </div>

          <div className="samples-row">
            {kaggleSamples.map((sample) => (
              <button
                key={sample.imagePath}
                className={`sample-card ${selectedSource?.kind === 'sample' && selectedSource.dataset === sample.dataset && selectedSource.imagePath === sample.imagePath ? 'selected' : ''}`}
                onClick={async () => await handleSampleSelect(sample)}
                type="button"
              >
                <img src={sample.imageUrl} alt={sample.fileName} />
                <div className="sample-caption">
                  <strong>{sample.className}</strong>
                  <span>{sample.fileName}</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        {error ? <div className="error-banner">{error}</div> : null}
      </main>
    </div>
  )
}

export default App

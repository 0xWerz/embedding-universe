"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface WordNode {
  id: string;
  text: string;
  embedding: number[];
  x: number;
  y: number;
  color: string;
  connections: string[];
  cluster?: number;
}

interface Connection {
  from: string;
  to: string;
  similarity: number;
}

interface ViewState {
  x: number;
  y: number;
  scale: number;
}

const COLORS = [
  "#8B5CF6", // Purple
  "#06B6D4", // Cyan
  "#10B981", // Emerald
  "#F59E0B", // Amber
  "#EF4444", // Red
  "#EC4899", // Pink
  "#6366F1", // Indigo
  "#84CC16", // Lime
  "#F97316", // Orange
  "#14B8A6", // Teal
  "#3B82F6", // Blue
  "#A855F7", // Violet
  "#22C55E", // Green
  "#F43F5E", // Rose
];

export default function EmbeddingPage() {
  const [inputText, setInputText] = useState("");
  const [words, setWords] = useState<WordNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isInitializingModel, setIsInitializingModel] = useState(false);

  // Cache the pipeline to avoid re-initialization
  const pipelineRef = useRef<any>(null);
  const [viewState, setViewState] = useState<ViewState>({
    x: 400, // Center horizontally (approximate)
    y: 300, // Center vertically (approximate)
    scale: 1,
  });

  // Initialize view state when container is available
  useEffect(() => {
    if (containerRef.current) {
      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;

      setViewState({
        x: containerWidth / 2,
        y: containerHeight / 2,
        scale: 1,
      });
    }
  }, []);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showClusters, setShowClusters] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const getEmbedding = async (text: string): Promise<number[]> => {
    const { pipeline } = await import("@huggingface/transformers");

    // Use cached pipeline if available
    if (!pipelineRef.current) {
      // Show initialization status
      setIsInitializingModel(true);

      // Create and cache embedding pipeline
      pipelineRef.current = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        {
          dtype: "q8", // Use 8-bit quantization for better performance
          device: "wasm", // Use WebAssembly for compatibility
        }
      );

      // Hide initialization status when done
      setIsInitializingModel(false);
    }

    // Get embeddings using cached pipeline
    const output = await pipelineRef.current(text, {
      pooling: "mean",
      normalize: true,
    });

    // Convert to flat array
    return Array.from(output.data);
  };

  const cosineSimilarity = (a: number[], b: number[]): number => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
  };

  // Force-directed layout for better positioning
  const generatePosition = useCallback(
    (embedding: number[], existingWords: WordNode[]) => {
      if (existingWords.length === 0) {
        // Start the first word at center (0, 0)
        return { x: 0, y: 0 };
      }

      // Calculate similarities and find the best position
      const similarities = existingWords.map((word) => ({
        word,
        similarity: cosineSimilarity(embedding, word.embedding),
      }));

      // Sort by similarity (highest first)
      similarities.sort((a, b) => b.similarity - a.similarity);

      // Find the most similar word
      const mostSimilar = similarities[0];

      if (mostSimilar.similarity > 0.2) {
        // Position based on similarity - closer = more similar
        // Map similarity (0.2-1.0) to distance (400-60)
        const normalizedSim = (mostSimilar.similarity - 0.2) / 0.8;
        const baseDistance = 400 - normalizedSim * 340; // 400 to 60 pixels

        // Add some angle variation for visual appeal
        const angle = Math.random() * Math.PI * 2;
        const distance = baseDistance + (Math.random() - 0.5) * 30;

        return {
          x: mostSimilar.word.x + Math.cos(angle) * distance,
          y: mostSimilar.word.y + Math.sin(angle) * distance,
        };
      }

      // If no strong similarity, position away from existing words
      const spread = Math.max(400, Math.sqrt(existingWords.length) * 250);
      return {
        x: (Math.random() - 0.5) * spread,
        y: (Math.random() - 0.5) * spread,
      };
    },
    []
  );

  const addWord = async (text: string) => {
    if (
      !text.trim() ||
      words.some((w) => w.text.toLowerCase() === text.toLowerCase())
    ) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const embedding = await getEmbedding(text);
      const position = generatePosition(embedding, words);
      const color = COLORS[words.length % COLORS.length];

      const newWord: WordNode = {
        id: Date.now().toString(),
        text: text,
        embedding,
        x: position.x,
        y: position.y,
        color,
        connections: [],
      };

      const updatedWords = [...words, newWord];
      const newConnections: Connection[] = [];

      // Calculate connections - show more relationships
      for (const existingWord of words) {
        const similarity = cosineSimilarity(embedding, existingWord.embedding);
        if (similarity > 0.2) {
          // Much lower threshold for better connectivity
          newConnections.push({
            from: existingWord.id,
            to: newWord.id,
            similarity,
          });
          existingWord.connections.push(newWord.id);
          newWord.connections.push(existingWord.id);
        }
      }

      setWords(updatedWords);
      setConnections((prev) => [...prev, ...newConnections]);
      setInputText("");

      // Auto-center view if this is the first word
      if (updatedWords.length === 1) {
        setTimeout(() => {
          resetView();
        }, 100);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add word");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setWords([]);
    setConnections([]);
    setHoveredWord(null);
    setHoveredConnection(null);
    setError("");
    resetView();
  };

  const getVisibleConnections = () => {
    // Always show medium+ connections, highlight when hovered
    if (!hoveredWord) {
      return connections.filter((conn) => conn.similarity > 0.3); // Show medium+ connections
    }

    // Show all connections for the hovered word
    return connections.filter(
      (conn) => conn.from === hoveredWord || conn.to === hoveredWord
    );
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) {
      addWord(inputText);
    }
  };

  // Mouse/touch handlers for pan and zoom
  const handleMouseDown = (e: React.MouseEvent) => {
    // Don't start dragging if clicking on a word bubble
    if (
      (e.target as Element).tagName === "circle" ||
      (e.target as Element).tagName === "text"
    ) {
      return;
    }
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - viewState.x, y: e.clientY - viewState.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setViewState((prev) => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      }));
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(3, viewState.scale * delta));

    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      setViewState((prev) => ({
        x: centerX - (centerX - prev.x) * (newScale / prev.scale),
        y: centerY - (centerY - prev.y) * (newScale / prev.scale),
        scale: newScale,
      }));
    }
  };

  const resetView = () => {
    if (words.length > 0) {
      const avgX = words.reduce((sum, w) => sum + w.x, 0) / words.length;
      const avgY = words.reduce((sum, w) => sum + w.y, 0) / words.length;

      const containerWidth = containerRef.current?.clientWidth || 800;
      const containerHeight = containerRef.current?.clientHeight || 600;

      setViewState({
        x: -avgX + containerWidth / 2,
        y: -avgY + containerHeight / 2,
        scale: 1,
      });
    } else {
      const containerWidth = containerRef.current?.clientWidth || 800;
      const containerHeight = containerRef.current?.clientHeight || 600;

      setViewState({
        x: containerWidth / 2,
        y: containerHeight / 2,
        scale: 1,
      });
    }
  };

  return (
    <div className="h-screen bg-black flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 bg-zinc-950/80 backdrop-blur-sm border-b border-zinc-800 p-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div>
            <h1 className="text-2xl font-bold text-zinc-100">
              Embedding Universe
            </h1>
            <p className="text-zinc-500 text-sm">
              Explore infinite semantic space
            </p>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowInstructions(!showInstructions)}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Help
            </button>
            <button
              onClick={resetView}
              className="bg-zinc-800 text-zinc-100 px-4 py-2 rounded-md hover:bg-zinc-700 transition-colors"
            >
              Reset View
            </button>
            <button
              onClick={clearAll}
              className="bg-red-900 text-zinc-100 px-4 py-2 rounded-md hover:bg-red-800 transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex relative">
        {/* Controls Panel */}
        <div className="w-80 bg-zinc-950/50 backdrop-blur-sm border-r border-zinc-800 p-4 flex flex-col gap-4">
          {/* Add Word */}
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <h3 className="text-zinc-200 font-medium mb-3">Add Word</h3>
            <div className="space-y-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-full p-2 bg-zinc-800 border border-zinc-700 rounded-md focus:ring-2 focus:ring-zinc-500 focus:border-zinc-500 text-zinc-100 placeholder-zinc-500"
                placeholder="Enter word..."
                disabled={loading}
              />
              <button
                onClick={() => addWord(inputText)}
                disabled={loading || !inputText.trim()}
                className="w-full bg-zinc-700 text-zinc-100 px-4 py-2 rounded-md hover:bg-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (isInitializingModel ? "Init..." : "...") : "Add"}
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <h3 className="text-zinc-200 font-medium mb-3">Stats</h3>
            <div className="text-zinc-300 text-sm space-y-1">
              <div>Words: {words.length}</div>
              <div>Connections: {connections.length}</div>
              <div>Zoom: {(viewState.scale * 100).toFixed(0)}%</div>
              {hoveredWord && (
                <div className="text-zinc-400">
                  Hovered: {words.find((w) => w.id === hoveredWord)?.text}
                </div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <h3 className="text-zinc-200 font-medium mb-3">Controls</h3>
            <div className="text-zinc-400 text-sm space-y-1">
              <div>• Hover to highlight connections</div>
              <div>• Drag to pan</div>
              <div>• Scroll to zoom</div>
              <div>• Press Enter to add word</div>
            </div>
          </div>

          {error && (
            <div className="bg-red-950/20 border border-red-800 rounded-lg p-4">
              <p className="text-red-400 text-sm">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Infinite Canvas */}
        <div
          ref={containerRef}
          className="flex-1 relative cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
        >
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            className="absolute inset-0"
            style={{
              cursor: isDragging ? "grabbing" : "grab",
              touchAction: "none",
            }}
          >
            {/* Background Grid */}
            <defs>
              <pattern
                id="grid"
                width="50"
                height="50"
                patternUnits="userSpaceOnUse"
                patternTransform={`scale(${viewState.scale}) translate(${
                  viewState.x / viewState.scale
                } ${viewState.y / viewState.scale})`}
              >
                <path
                  d="M 50 0 L 0 0 0 50"
                  fill="none"
                  stroke="rgba(161, 161, 170, 0.1)"
                  strokeWidth="1"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            {/* Transformed content */}
            <g
              transform={`translate(${viewState.x} ${viewState.y}) scale(${viewState.scale})`}
            >
              {/* Connection Lines */}
              {getVisibleConnections().map((connection, index) => {
                const fromWord = words.find((w) => w.id === connection.from);
                const toWord = words.find((w) => w.id === connection.to);

                if (!fromWord || !toWord) return null;

                const connectionId = `${connection.from}-${connection.to}`;
                const isHighlighted =
                  hoveredWord === connection.from ||
                  hoveredWord === connection.to;
                const isHovered = hoveredConnection === connectionId;

                // Scale visual properties based on similarity
                const similarity = connection.similarity;
                const lineWidth = Math.max(1, similarity * 6);
                const opacity = Math.max(0.3, similarity * 0.8);

                // Color based on similarity strength
                const getConnectionColor = (sim: number) => {
                  if (sim > 0.7) return "#10b981"; // Strong - emerald
                  if (sim > 0.5) return "#3b82f6"; // Medium - blue
                  if (sim > 0.3) return "#f59e0b"; // Weak - amber
                  return "#64748b"; // Very weak - gray
                };

                return (
                  <g key={index}>
                    <line
                      x1={fromWord.x}
                      y1={fromWord.y}
                      x2={toWord.x}
                      y2={toWord.y}
                      stroke={
                        isHighlighted || isHovered
                          ? "#ffffff"
                          : getConnectionColor(similarity)
                      }
                      strokeWidth={
                        isHighlighted || isHovered ? lineWidth + 2 : lineWidth
                      }
                      strokeOpacity={isHighlighted || isHovered ? 1 : opacity}
                      className="transition-all duration-300 cursor-pointer"
                      onMouseEnter={() => setHoveredConnection(connectionId)}
                      onMouseLeave={() => setHoveredConnection(null)}
                    />
                    {/* Always show percentage for strong connections or when highlighted/hovered */}
                    {(similarity > 0.5 || isHighlighted || isHovered) && (
                      <g>
                        {/* Background for better readability */}
                        <rect
                          x={(fromWord.x + toWord.x) / 2 - 15}
                          y={(fromWord.y + toWord.y) / 2 - 18}
                          width="30"
                          height="16"
                          fill="rgba(0, 0, 0, 0.8)"
                          rx="8"
                          className="pointer-events-none"
                        />
                        <text
                          x={(fromWord.x + toWord.x) / 2}
                          y={(fromWord.y + toWord.y) / 2 - 8}
                          fill={
                            isHighlighted || isHovered ? "#ffffff" : "#e4e4e7"
                          }
                          fontSize={isHighlighted || isHovered ? "13" : "11"}
                          fontWeight="600"
                          textAnchor="middle"
                          className="font-mono transition-all duration-300 pointer-events-none"
                          style={{
                            filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.4))",
                          }}
                        >
                          {(similarity * 100).toFixed(0)}%
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}

              {/* Word Bubbles */}
              {words.map((word) => {
                const isHovered = hoveredWord === word.id;
                const isConnected =
                  hoveredWord && word.connections.includes(hoveredWord);
                const opacity =
                  !hoveredWord || isHovered || isConnected ? 1 : 0.25;
                const scale = isHovered ? 1.15 : 1;

                return (
                  <g key={word.id}>
                    {/* Glow effect when hovered */}
                    {isHovered && (
                      <circle
                        cx={word.x}
                        cy={word.y}
                        r={35 * scale}
                        fill={word.color}
                        fillOpacity={0.2}
                        className="transition-all duration-300"
                      />
                    )}
                    <circle
                      cx={word.x}
                      cy={word.y}
                      r={28 * scale}
                      fill={word.color}
                      fillOpacity={opacity}
                      stroke={
                        isHovered
                          ? "#ffffff"
                          : isConnected
                          ? "#ffffff"
                          : word.color
                      }
                      strokeWidth={isHovered ? 2.5 : isConnected ? 2 : 1.5}
                      strokeOpacity={isHovered || isConnected ? 1 : 0.7}
                      className="transition-all duration-300 cursor-pointer"
                      onMouseEnter={() => setHoveredWord(word.id)}
                      onMouseLeave={() => setHoveredWord(null)}
                      style={{
                        filter: isHovered
                          ? "drop-shadow(0 0 12px rgba(255,255,255,0.5))"
                          : isConnected
                          ? "drop-shadow(0 0 8px rgba(255,255,255,0.3))"
                          : "drop-shadow(0 2px 4px rgba(0,0,0,0.3))",
                      }}
                    />
                    <text
                      x={word.x}
                      y={word.y}
                      fill="white"
                      fontSize={isHovered ? "14" : "12"}
                      fontWeight="700"
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="pointer-events-none select-none transition-all duration-300"
                      fillOpacity={opacity}
                      style={{
                        filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.8))",
                      }}
                    >
                      {word.text.length > 10
                        ? word.text.substring(0, 10) + "..."
                        : word.text}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>

          {/* Instructions Overlay */}
          {showInstructions && words.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="bg-zinc-900/95 backdrop-blur-md rounded-2xl p-10 max-w-xl text-center border border-zinc-800/50 shadow-2xl">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-zinc-100 mb-4">
                  Embedding Universe
                </h3>
                <p className="text-zinc-300 mb-6 leading-relaxed">
                  Explore how AI understands language by building an interactive
                  semantic network. Powered by Transformers.js, this runs
                  entirely in your browser with automatic fallback to Ollama for
                  enhanced performance.
                </p>
                <div className="grid grid-cols-2 gap-4 text-zinc-400 text-sm mb-8">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <span>Browser-based AI</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span>Live connections</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                    <span>Hover to explore</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                    <span>Distance = similarity</span>
                  </div>
                </div>
                <div className="text-zinc-500 text-sm">
                  Add your first word to begin exploring →
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import React from 'react';

const BERTLacunaComparison = () => {
  const Box = ({ children, color = "#3b82f6", small = false }) => (
    <div className={`border-2 rounded ${small ? 'px-2 py-1 text-xs' : 'px-4 py-2 text-sm'} text-center font-medium`}
         style={{ borderColor: color, backgroundColor: `${color}15` }}>
      {children}
    </div>
  );
  
  const Arrow = ({ label = "" }) => (
    <div className="flex flex-col items-center justify-center py-1">
      <div className="text-lg text-gray-600">↓</div>
      {label && <div className="text-xs text-gray-500 italic">{label}</div>}
    </div>
  );

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50">
      <h1 className="text-3xl font-bold text-center mb-2">
        BERT vs Lacuna: Architectural Correspondence
      </h1>
      <p className="text-center text-sm text-gray-600 mb-6">
        How Lacuna adapts BERT's transformer architecture for tabular missing data
      </p>

      <div className="grid grid-cols-2 gap-6">
        {/* LEFT COLUMN: BERT */}
        <div className="space-y-3">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-blue-600 mb-1">BERT</h2>
            <p className="text-xs text-gray-600">Masked Language Modeling</p>
          </div>

          {/* BERT Input */}
          <div className="bg-white p-3 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-sm mb-2 text-center text-blue-700">INPUT</div>
            <div className="text-xs mb-2 text-center text-gray-600">Sentence with masked word</div>
            <div className="flex justify-center gap-1 flex-wrap">
              <Box color="#3b82f6" small>The</Box>
              <Box color="#3b82f6" small>cat</Box>
              <Box color="#ef4444" small>[MASK]</Box>
              <Box color="#3b82f6" small>on</Box>
              <Box color="#3b82f6" small>the</Box>
              <Box color="#3b82f6" small>mat</Box>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2">
              Sequence of discrete tokens
            </div>
          </div>

          <Arrow label="Token IDs" />

          {/* BERT Embedding */}
          <div className="bg-white p-3 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-sm mb-2 text-center text-blue-700">TOKEN EMBEDDING</div>
            <div className="space-y-2">
              <div className="text-xs">
                <Box color="#3b82f6" small>Token Embedding (vocab lookup)</Box>
                <div className="text-center text-xs text-gray-500">+</div>
                <Box color="#8b5cf6" small>Position Embedding</Box>
                <div className="text-center text-xs text-gray-500">+</div>
                <Box color="#06b6d4" small>Segment Embedding</Box>
              </div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [seq_len, hidden_dim]
            </div>
          </div>

          <Arrow label="Embedded sequence" />

          {/* BERT Transformer */}
          <div className="bg-white p-3 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-sm mb-2 text-center text-blue-700">TRANSFORMER ENCODER</div>
            <div className="space-y-2 text-xs">
              <div className="p-2 bg-blue-50 rounded">
                <div className="font-semibold mb-1">Multi-Head Self-Attention</div>
                <div className="text-xs text-gray-600">Each word attends to ALL words</div>
                <div className="flex justify-center gap-1 mt-1">
                  <Box color="#6366f1" small>Q</Box>
                  <Box color="#6366f1" small>K</Box>
                  <Box color="#6366f1" small>V</Box>
                </div>
              </div>
              <div className="text-center text-gray-500">↓ Add & Norm</div>
              <div className="p-2 bg-blue-50 rounded">
                <div className="font-semibold mb-1">Feed-Forward Network</div>
                <div className="text-xs text-gray-600">Position-wise FFN</div>
              </div>
              <div className="text-center text-gray-500">↓ Add & Norm</div>
              <div className="text-center font-bold text-blue-600">× 12 Layers</div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [seq_len, hidden_dim]
            </div>
          </div>

          <Arrow label="Contextualized representations" />

          {/* BERT Pooling */}
          <div className="bg-white p-3 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-sm mb-2 text-center text-blue-700">SEQUENCE REPRESENTATION</div>
            <div className="space-y-2">
              <div className="text-xs text-center mb-1">Use [CLS] token</div>
              <Box color="#1e40af">[CLS] representation</Box>
              <div className="text-xs text-center text-gray-500 mt-1">
                Special token at sequence start
              </div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [hidden_dim]
            </div>
          </div>

          <Arrow label="Sentence embedding" />

          {/* BERT Output */}
          <div className="bg-white p-3 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-sm mb-2 text-center text-blue-700">OUTPUT HEAD</div>
            <div className="space-y-2">
              <Box color="#3b82f6">Linear (hidden → vocab_size)</Box>
              <Arrow />
              <Box color="#1e40af">Softmax over vocabulary</Box>
              <div className="text-xs text-center mt-2 font-semibold">
                Predicted word: "sat" (98%)
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: LACUNA */}
        <div className="space-y-3">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-purple-600 mb-1">Lacuna</h2>
            <p className="text-xs text-gray-600">Missing Mechanism Classification</p>
          </div>

          {/* Lacuna Input */}
          <div className="bg-white p-3 rounded-lg border-2 border-purple-300">
            <div className="font-bold text-sm mb-2 text-center text-purple-700">INPUT</div>
            <div className="text-xs mb-2 text-center text-gray-600">Row with missing feature</div>
            <div className="flex justify-center gap-1 flex-wrap">
              <Box color="#9333ea" small>F1: 2.3</Box>
              <Box color="#ef4444" small>F2: NaN</Box>
              <Box color="#9333ea" small>F3: -1.2</Box>
              <Box color="#9333ea" small>F4: 0.8</Box>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2">
              Row of continuous values + missing
            </div>
          </div>

          <Arrow label="4D cell tokens" />

          {/* Lacuna Embedding */}
          <div className="bg-white p-3 rounded-lg border-2 border-purple-300">
            <div className="font-bold text-sm mb-2 text-center text-purple-700">TOKEN EMBEDDING</div>
            <div className="space-y-2">
              <div className="text-xs">
                <Box color="#9333ea" small>Value Embedding (linear proj)</Box>
                <div className="text-center text-xs text-gray-500">+</div>
                <Box color="#a855f7" small>Observation Embedding (lookup)</Box>
                <div className="text-center text-xs text-gray-500">+</div>
                <Box color="#c026d3" small>Mask Type Embedding (lookup)</Box>
                <div className="text-center text-xs text-gray-500">+</div>
                <Box color="#7c3aed" small>Feature Position Embedding</Box>
              </div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [n_rows, n_features, hidden_dim]
            </div>
          </div>

          <Arrow label="Embedded features" />

          {/* Lacuna Transformer */}
          <div className="bg-white p-3 rounded-lg border-2 border-purple-300">
            <div className="font-bold text-sm mb-2 text-center text-purple-700">TRANSFORMER ENCODER</div>
            <div className="space-y-2 text-xs">
              <div className="p-2 bg-purple-50 rounded">
                <div className="font-semibold mb-1">Multi-Head Self-Attention</div>
                <div className="text-xs text-gray-600">Each feature attends to features IN SAME ROW</div>
                <div className="flex justify-center gap-1 mt-1">
                  <Box color="#8b5cf6" small>Q</Box>
                  <Box color="#8b5cf6" small>K</Box>
                  <Box color="#8b5cf6" small>V</Box>
                </div>
              </div>
              <div className="text-center text-gray-500">↓ Add & Norm</div>
              <div className="p-2 bg-purple-50 rounded">
                <div className="font-semibold mb-1">Feed-Forward Network</div>
                <div className="text-xs text-gray-600">Position-wise FFN</div>
              </div>
              <div className="text-center text-gray-500">↓ Add & Norm</div>
              <div className="text-center font-bold text-purple-600">× 4 Layers</div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [n_rows, n_features, hidden_dim]
            </div>
          </div>

          <Arrow label="Contextualized features" />

          {/* Lacuna Pooling */}
          <div className="bg-white p-3 rounded-lg border-2 border-purple-300">
            <div className="font-bold text-sm mb-2 text-center text-purple-700">HIERARCHICAL POOLING</div>
            <div className="space-y-2">
              <div className="text-xs">
                <div className="text-center font-semibold mb-1">Row Pooling</div>
                <Box color="#7c3aed" small>Attention over features → row repr</Box>
              </div>
              <div className="text-center text-gray-500">↓</div>
              <div className="text-xs">
                <div className="text-center font-semibold mb-1">Dataset Pooling</div>
                <Box color="#6d28d9" small>Attention over rows → dataset repr</Box>
              </div>
              <div className="text-xs text-center text-gray-500 mt-1">
                No [CLS] token - learned pooling
              </div>
            </div>
            <div className="text-xs text-center text-gray-500 mt-2 font-mono">
              → [evidence_dim]
            </div>
          </div>

          <Arrow label="Evidence vector" />

          {/* Lacuna Output */}
          <div className="bg-white p-3 rounded-lg border-2 border-purple-300">
            <div className="font-bold text-sm mb-2 text-center text-purple-700">OUTPUT HEAD</div>
            <div className="space-y-2">
              <Box color="#9333ea">Linear (evidence → 110 generators)</Box>
              <Arrow />
              <Box color="#7c3aed">Softmax → aggregate to 3 classes</Box>
              <div className="text-xs text-center mt-2 font-semibold">
                MAR: 72%, MCAR: 15%, MNAR: 13%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison Summary */}
      <div className="mt-8 bg-white p-6 rounded-lg border-2 border-gray-300">
        <h2 className="text-xl font-bold text-center mb-4">Architectural Correspondence & Divergence</h2>
        
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="p-3 bg-green-50 rounded border border-green-300">
            <div className="font-bold text-green-800 mb-2">✓ Shared Design</div>
            <ul className="text-xs space-y-1">
              <li>• Token-based input representation</li>
              <li>• Self-attention over sequences</li>
              <li>• Bidirectional context (attend all)</li>
              <li>• Multi-head attention mechanism</li>
              <li>• Position embeddings</li>
              <li>• Stacked transformer layers</li>
              <li>• Pre-norm architecture</li>
              <li>• Residual connections</li>
            </ul>
          </div>
          
          <div className="p-3 bg-yellow-50 rounded border border-yellow-300">
            <div className="font-bold text-yellow-800 mb-2">⚠ Adapted for Tabular</div>
            <ul className="text-xs space-y-1">
              <li>• 4D tokens vs vocab lookup</li>
              <li>• Continuous values + binary flags</li>
              <li>• Row-wise attention scope</li>
              <li>• No segment embeddings</li>
              <li>• Hierarchical pooling vs [CLS]</li>
              <li>• Fewer layers (4 vs 12)</li>
              <li>• Explicit missingness encoding</li>
            </ul>
          </div>
          
          <div className="p-3 bg-blue-50 rounded border border-blue-300">
            <div className="font-bold text-blue-800 mb-2">🎯 Novel Contributions</div>
            <ul className="text-xs space-y-1">
              <li>• Mask type for self-supervision</li>
              <li>• Two-stage attention pooling</li>
              <li>• Generator-to-class aggregation</li>
              <li>• Calibrated posteriors (not logits)</li>
              <li>• Multiple signal integration</li>
              <li>• Row-independence assumption</li>
              <li>• Tabular-specific inductive bias</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gradient-to-r from-blue-100 to-purple-100 rounded">
          <div className="font-bold text-center mb-2">Core Insight: Why BERT Architecture Works Here</div>
          <div className="text-sm text-center">
            <strong>BERT's task:</strong> Predict masked word from sentence context via cross-word dependencies
            <br/>
            <strong>Lacuna's task:</strong> Infer why data is missing from feature context via cross-column dependencies
            <br/><br/>
            <span className="text-purple-700 font-semibold">
              MAR = "missingness in column A depends on values in column B" 
              <br/>
              → This IS a cross-column dependency that self-attention can learn!
            </span>
          </div>
        </div>
      </div>

      {/* Key Architectural Decisions */}
      <div className="mt-6 bg-gradient-to-r from-purple-100 to-pink-100 p-6 rounded-lg border-2 border-purple-300">
        <h2 className="text-xl font-bold text-center mb-4">Why Each Design Decision Matters</h2>
        
        <div className="space-y-3 text-sm">
          <div className="p-3 bg-white rounded">
            <div className="font-bold text-purple-700">1. Why 4D tokens (not just embeddings)?</div>
            <div className="text-xs mt-1">
              <strong>BERT:</strong> "cat" → single embedding vector (the word IS its representation)
              <br/>
              <strong>Lacuna:</strong> Cell has FOUR orthogonal properties: its value, whether it's observed, whether it's artificially masked, and which feature it is. 
              Separating these enables learning: "high values go missing" (MNAR) vs "missingness ignores values" (MCAR).
            </div>
          </div>

          <div className="p-3 bg-white rounded">
            <div className="font-bold text-purple-700">2. Why row-wise attention (not full-dataset)?</div>
            <div className="text-xs mt-1">
              <strong>Mechanism Theory:</strong> MAR means P(R_ij | X_obs) ≠ P(R_ij) — missingness depends on OTHER variables IN THE SAME ROW.
              <br/>
              Cross-row attention would capture "Feature 1 is 30% missing across the dataset" (a summary statistic) but NOT "Feature 1 is missing WHEN Feature 2 > 5 in this row" (MAR signal).
              <br/>
              <strong>Bonus:</strong> O(d²) memory per row vs O(n²d²) for full dataset → enables larger datasets.
            </div>
          </div>

          <div className="p-3 bg-white rounded">
            <div className="font-bold text-purple-700">3. Why hierarchical pooling (not [CLS] token)?</div>
            <div className="text-xs mt-1">
              <strong>BERT:</strong> Sentence has natural start position → [CLS] token there
              <br/>
              <strong>Lacuna:</strong> Tabular data has no privileged start position. Which row/feature should [CLS] attend to first?
              <br/>
              Attention pooling learns: "For MAR detection, rows with partial missingness are most informative. For MNAR, rows with extreme values matter more."
              This is learned, not assumed.
            </div>
          </div>

          <div className="p-3 bg-white rounded">
            <div className="font-bold text-purple-700">4. Why generator posteriors (not direct 3-class)?</div>
            <div className="text-xs mt-1">
              <strong>Problem:</strong> If we train on 5 MCAR generators directly, model might learn "Generator 3 always sets 20% missing in column 5" (fingerprints).
              <br/>
              <strong>Solution:</strong> Train on 110+ diverse generators → forces learning "this PATTERN looks MCAR-like" not "this is Generator 3."
              <br/>
              Aggregation provides interpretable output while preventing overfitting to implementation details.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BERTLacunaComparison;
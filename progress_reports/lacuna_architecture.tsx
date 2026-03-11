import React from 'react';

const LacunaArchitectureDiagram = () => {
  // Reusable components
  const Box = ({ children, color = "blue", className = "" }) => (
    <div className={`border-2 rounded px-3 py-2 text-center text-sm font-medium ${className}`}
         style={{ borderColor: color, backgroundColor: `${color}15` }}>
      {children}
    </div>
  );
  
  const Arrow = ({ label = "", vertical = false }) => (
    <div className={`flex ${vertical ? 'flex-col' : 'flex-row'} items-center justify-center`}>
      {vertical ? (
        <>
          <div className="text-xl text-gray-600">↓</div>
          {label && <div className="text-xs text-gray-600 font-semibold">{label}</div>}
        </>
      ) : (
        <>
          <div className="text-xl text-gray-600">→</div>
          {label && <div className="text-xs text-gray-600 font-semibold ml-1">{label}</div>}
        </>
      )}
    </div>
  );

  const Dimension = ({ text }) => (
    <div className="text-xs text-blue-600 font-mono mt-1">{text}</div>
  );

  return (
    <div className="w-full max-w-7xl mx-auto p-8 bg-white">
      <h1 className="text-2xl font-bold text-center mb-6">
        Lacuna Architecture: BERT-Style Network Diagram
      </h1>
      
      {/* INPUT LAYER */}
      <div className="mb-4">
        <div className="text-center text-lg font-bold text-purple-700 mb-2">INPUT LAYER</div>
        <div className="text-center text-sm text-gray-600 mb-3">Example: Single row with 4 features</div>
        
        <div className="flex justify-center gap-2 mb-2">
          <Box color="#9333ea">
            <div className="font-bold">Feature 1</div>
            <div className="text-xs">[2.3, 1, 0, 0.00]</div>
            <div className="text-xs text-gray-500">val, obs, mask, id</div>
          </Box>
          <Box color="#9333ea">
            <div className="font-bold">Feature 2</div>
            <div className="text-xs">[0.0, 0, 0, 0.33]</div>
            <div className="text-xs text-gray-500">MISSING</div>
          </Box>
          <Box color="#9333ea">
            <div className="font-bold">Feature 3</div>
            <div className="text-xs">[-1.2, 1, 0, 0.67]</div>
            <div className="text-xs text-gray-500">val, obs, mask, id</div>
          </Box>
          <Box color="#9333ea">
            <div className="font-bold">Feature 4</div>
            <div className="text-xs">[0.8, 1, 0, 1.00]</div>
            <div className="text-xs text-gray-500">val, obs, mask, id</div>
          </Box>
        </div>
        <Dimension text="Shape: [B, n_rows, n_features, 4]" />
        <div className="text-xs text-center text-gray-500 mt-1">
          Each cell is a 4D token: [value, is_observed, mask_type, feature_id]
        </div>
      </div>

      <Arrow vertical label="Tokenization complete" />

      {/* TOKEN EMBEDDING LAYER */}
      <div className="mb-4 bg-blue-50 p-4 rounded-lg">
        <div className="text-center text-lg font-bold text-blue-700 mb-3">TOKEN EMBEDDING LAYER</div>
        
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="text-center">
            <div className="text-xs font-semibold mb-2">Feature 1</div>
            <div className="space-y-1">
              <Box color="#3b82f6" className="text-xs">Value Proj<br/>Linear(2.3×1)</Box>
              <Box color="#3b82f6" className="text-xs">Obs Embed<br/>lookup[1]</Box>
              <Box color="#3b82f6" className="text-xs">Mask Embed<br/>lookup[0]</Box>
              <Box color="#3b82f6" className="text-xs">Pos Embed<br/>lookup[0]</Box>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-xs font-semibold mb-2">Feature 2</div>
            <div className="space-y-1">
              <Box color="#3b82f6" className="text-xs">Value Proj<br/>Linear(0.0×0)</Box>
              <Box color="#ef4444" className="text-xs font-bold">Obs Embed<br/>lookup[0]</Box>
              <Box color="#3b82f6" className="text-xs">Mask Embed<br/>lookup[0]</Box>
              <Box color="#3b82f6" className="text-xs">Pos Embed<br/>lookup[1]</Box>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-xs font-semibold mb-2">Feature 3</div>
            <div className="space-y-1">
              <Box color="#3b82f6" className="text-xs">Value Proj<br/>Linear(-1.2×1)</Box>
              <Box color="#3b82f6" className="text-xs">Obs Embed<br/>lookup[1]</Box>
              <Box color="#3b82f6" className="text-xs">Mask Embed<br/>lookup[0]</Box>
              <Box color="#3b82f6" className="text-xs">Pos Embed<br/>lookup[2]</Box>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-xs font-semibold mb-2">Feature 4</div>
            <div className="space-y-1">
              <Box color="#3b82f6" className="text-xs">Value Proj<br/>Linear(0.8×1)</Box>
              <Box color="#3b82f6" className="text-xs">Obs Embed<br/>lookup[1]</Box>
              <Box color="#3b82f6" className="text-xs">Mask Embed<br/>lookup[0]</Box>
              <Box color="#3b82f6" className="text-xs">Pos Embed<br/>lookup[3]</Box>
            </div>
          </div>
        </div>
        
        <Arrow vertical label="Concatenate [h/4, h/4, h/4, h/4] → h" />
        
        <div className="flex justify-center gap-2 mt-3">
          <Box color="#1e40af" className="px-6">Feat 1 Embedding<br/>(h=128 dims)</Box>
          <Box color="#dc2626" className="px-6">Feat 2 Embedding<br/>(h=128 dims)</Box>
          <Box color="#1e40af" className="px-6">Feat 3 Embedding<br/>(h=128 dims)</Box>
          <Box color="#1e40af" className="px-6">Feat 4 Embedding<br/>(h=128 dims)</Box>
        </div>
        <Dimension text="Shape: [B, n_rows, n_features, h=128]" />
      </div>

      <Arrow vertical label="Pass through LayerNorm + Dropout" />

      {/* TRANSFORMER ENCODER LAYER 1 */}
      <div className="mb-4 bg-indigo-50 p-4 rounded-lg border-2 border-indigo-300">
        <div className="text-center text-lg font-bold text-indigo-700 mb-3">
          TRANSFORMER LAYER 1 (of 4 layers)
        </div>
        <div className="text-xs text-center text-gray-600 mb-3">
          Row-wise self-attention: Each row processed independently
        </div>
        
        {/* Multi-Head Attention */}
        <div className="mb-4 p-3 bg-white rounded border border-indigo-200">
          <div className="text-sm font-bold text-center mb-3">Multi-Head Self-Attention (4 heads)</div>
          
          <div className="flex justify-center gap-3 mb-3">
            <div className="text-center">
              <Box color="#6366f1" className="mb-1">Linear<sub>Q</sub></Box>
              <div className="text-xs font-bold">Queries (Q)</div>
              <Dimension text="[4, 128, 32]" />
            </div>
            <div className="text-center">
              <Box color="#6366f1" className="mb-1">Linear<sub>K</sub></Box>
              <div className="text-xs font-bold">Keys (K)</div>
              <Dimension text="[4, 128, 32]" />
            </div>
            <div className="text-center">
              <Box color="#6366f1" className="mb-1">Linear<sub>V</sub></Box>
              <div className="text-xs font-bold">Values (V)</div>
              <Dimension text="[4, 128, 32]" />
            </div>
          </div>
          
          <Arrow vertical label="" />
          
          <div className="text-center my-2">
            <Box color="#8b5cf6" className="inline-block px-6">
              Attention(Q,K,V) = softmax(QK<sup>T</sup>/√d<sub>k</sub>) V
            </Box>
          </div>
          
          <Arrow vertical label="" />
          
          <div className="grid grid-cols-4 gap-2 mb-2">
            <div className="text-xs text-center p-2 bg-purple-100 rounded border border-purple-300">
              <div className="font-bold mb-1">Head 1</div>
              <div className="text-xs">Feat1 attends to:<br/>F1: 0.4, F2: 0.1<br/>F3: 0.3, F4: 0.2</div>
            </div>
            <div className="text-xs text-center p-2 bg-purple-100 rounded border border-purple-300">
              <div className="font-bold mb-1">Head 2</div>
              <div className="text-xs">Feat2 attends to:<br/>F1: <strong>0.6</strong>, F2: 0.1<br/>F3: 0.2, F4: 0.1</div>
            </div>
            <div className="text-xs text-center p-2 bg-purple-100 rounded border border-purple-300">
              <div className="font-bold mb-1">Head 3</div>
              <div className="text-xs">Different pattern<br/>learning different<br/>dependencies</div>
            </div>
            <div className="text-xs text-center p-2 bg-purple-100 rounded border border-purple-300">
              <div className="font-bold mb-1">Head 4</div>
              <div className="text-xs">Different pattern<br/>learning different<br/>dependencies</div>
            </div>
          </div>
          <div className="text-xs text-center text-red-600 font-semibold">
            ⚠ Notice: Missing Feature 2 strongly attends to observed Feature 1 (MAR signal!)
          </div>
          
          <Arrow vertical label="Concat heads + Linear projection" />
          
          <div className="flex justify-center gap-2 mt-2">
            <Box color="#4c1d95" className="px-4">Updated F1</Box>
            <Box color="#4c1d95" className="px-4">Updated F2</Box>
            <Box color="#4c1d95" className="px-4">Updated F3</Box>
            <Box color="#4c1d95" className="px-4">Updated F4</Box>
          </div>
          <Dimension text="[B, n_rows, n_features, h=128]" />
        </div>
        
        <Arrow vertical label="Residual + LayerNorm" />
        
        {/* Feed-Forward Network */}
        <div className="p-3 bg-white rounded border border-indigo-200">
          <div className="text-sm font-bold text-center mb-2">Position-wise Feed-Forward Network</div>
          
          <div className="flex justify-center items-center gap-3">
            <Box color="#6366f1">Input<br/>(h=128)</Box>
            <Arrow />
            <Box color="#8b5cf6">Linear<br/>(128→512)</Box>
            <Arrow />
            <Box color="#a855f7">ReLU</Box>
            <Arrow />
            <Box color="#8b5cf6">Linear<br/>(512→128)</Box>
            <Arrow />
            <Box color="#6366f1">Output<br/>(h=128)</Box>
          </div>
          
          <div className="text-xs text-center text-gray-600 mt-2">
            Applied independently to each feature token
          </div>
        </div>
        
        <Arrow vertical label="Residual + LayerNorm" />
        
        <div className="text-center mt-2">
          <Box color="#4338ca" className="inline-block px-8">
            Transformer Layer 1 Output
          </Box>
          <Dimension text="[B, n_rows, n_features, h=128]" />
        </div>
      </div>

      <div className="text-center my-3 text-sm font-bold text-indigo-600">
        ⟱ Repeat 3 more times (Layers 2, 3, 4) ⟱
      </div>

      {/* AFTER ALL TRANSFORMER LAYERS */}
      <div className="mb-4 bg-teal-50 p-4 rounded-lg border-2 border-teal-300">
        <div className="text-center text-lg font-bold text-teal-700 mb-2">
          HIERARCHICAL POOLING
        </div>
        
        {/* Row Pooling */}
        <div className="mb-4 p-3 bg-white rounded border border-teal-200">
          <div className="text-sm font-bold text-center mb-2">Step 1: Row Pooling (Features → Row)</div>
          
          <div className="flex justify-center items-center gap-2 mb-2">
            <Box color="#14b8a6" className="px-4">Feat 1<br/>(128d)</Box>
            <Box color="#14b8a6" className="px-4">Feat 2<br/>(128d)</Box>
            <Box color="#14b8a6" className="px-4">Feat 3<br/>(128d)</Box>
            <Box color="#14b8a6" className="px-4">Feat 4<br/>(128d)</Box>
          </div>
          
          <Arrow vertical label="" />
          
          <div className="text-center mb-2">
            <Box color="#0d9488" className="inline-block px-6">
              Attention Pooling<br/>
              <span className="text-xs">weights = softmax(Q<sub>row</sub> · Features<sup>T</sup>)</span>
            </Box>
          </div>
          
          <Arrow vertical label="Weighted sum" />
          
          <div className="flex justify-center">
            <Box color="#0f766e" className="px-8">
              Single Row Representation
            </Box>
          </div>
          <Dimension text="[B, n_rows, h=128]" />
          <div className="text-xs text-center text-gray-600 mt-1">
            All features in this row compressed to single vector
          </div>
        </div>
        
        {/* Dataset Pooling */}
        <div className="p-3 bg-white rounded border border-teal-200">
          <div className="text-sm font-bold text-center mb-2">Step 2: Dataset Pooling (Rows → Dataset)</div>
          
          <div className="flex justify-center items-center gap-2 mb-2">
            <Box color="#14b8a6" className="px-4">Row 1<br/>(128d)</Box>
            <Box color="#14b8a6" className="px-4">Row 2<br/>(128d)</Box>
            <div className="text-xl">...</div>
            <Box color="#14b8a6" className="px-4">Row n<br/>(128d)</Box>
          </div>
          
          <Arrow vertical label="" />
          
          <div className="text-center mb-2">
            <Box color="#0d9488" className="inline-block px-6">
              Attention Pooling + Linear Projection<br/>
              <span className="text-xs">weights = softmax(Q<sub>dataset</sub> · Rows<sup>T</sup>)</span>
            </Box>
          </div>
          
          <Arrow vertical label="Weighted sum + project to evidence_dim" />
          
          <div className="flex justify-center">
            <Box color="#0f766e" className="px-12 py-3">
              <div className="font-bold text-lg">Evidence Vector Z</div>
              <div className="text-xs">Complete dataset summary</div>
            </Box>
          </div>
          <Dimension text="[B, p=64]" />
        </div>
      </div>

      <Arrow vertical label="LayerNorm" />

      {/* CLASSIFICATION HEAD */}
      <div className="bg-green-50 p-4 rounded-lg border-2 border-green-300">
        <div className="text-center text-lg font-bold text-green-700 mb-3">
          CLASSIFICATION HEAD
        </div>
        
        <div className="flex justify-center items-center gap-3 mb-3">
          <Box color="#22c55e" className="px-8 py-3">
            Evidence Z<br/>(64 dims)
          </Box>
          <Arrow />
          <Box color="#16a34a" className="px-6">
            Linear<br/>(64 → 110)
          </Box>
          <Arrow />
          <Box color="#15803d" className="px-6">
            Softmax
          </Box>
        </div>
        
        <div className="flex justify-center mb-3">
          <Box color="#166534" className="px-12 py-2">
            <div className="font-bold">Generator Posterior p(g|Z)</div>
            <div className="text-xs mt-1">[g₁: 0.002, g₂: 0.001, ..., g₁₁₀: 0.015]</div>
          </Box>
        </div>
        
        <Arrow vertical label="Marginalize over generators" />
        
        <div className="text-center mb-2">
          <Box color="#14532d" className="inline-block px-8 py-3">
            <div className="font-bold text-lg">Class Posterior π(c|Z)</div>
            <div className="text-xs mt-1">p(MCAR) = Σ p(g<sub>MCAR</sub>)</div>
            <div className="text-xs">p(MAR) = Σ p(g<sub>MAR</sub>)</div>
            <div className="text-xs">p(MNAR) = Σ p(g<sub>MNAR</sub>)</div>
          </Box>
        </div>
        
        <Arrow vertical />
        
        <div className="grid grid-cols-3 gap-3">
          <Box color="#dc2626" className="py-3">
            <div className="font-bold">MCAR</div>
            <div className="text-lg">0.15</div>
          </Box>
          <Box color="#16a34a" className="py-3">
            <div className="font-bold">MAR</div>
            <div className="text-lg">0.72</div>
          </Box>
          <Box color="#2563eb" className="py-3">
            <div className="font-bold">MNAR</div>
            <div className="text-lg">0.13</div>
          </Box>
        </div>
        
        <div className="text-center mt-3 text-sm font-semibold text-green-800">
          ✓ Calibrated posterior probabilities (ECE = 0.042)
        </div>
      </div>

      {/* KEY INSIGHTS FOOTER */}
      <div className="mt-6 p-4 bg-gray-100 rounded-lg">
        <div className="text-center font-bold text-lg mb-2">Key Architectural Decisions</div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="p-2 bg-blue-50 rounded">
            <strong>4D Token Structure:</strong> Explicitly separates value, observation status, mask type, and position—enabling the model to learn independent patterns for each aspect rather than entangled representations
          </div>
          <div className="p-2 bg-indigo-50 rounded">
            <strong>Row-wise Attention:</strong> MAR manifests within rows (missingness in col A depends on values in col B <em>within same row</em>). Cross-row attention adds no signal but O(n²) memory cost
          </div>
          <div className="p-2 bg-teal-50 rounded">
            <strong>Hierarchical Pooling:</strong> Two-stage compression (features→rows→dataset) respects tabular structure. Attention weights learn which features/rows are most informative for mechanism classification
          </div>
          <div className="p-2 bg-green-50 rounded">
            <strong>Generator Posteriors:</strong> Training on 110+ fine-grained mechanisms prevents "fingerprint learning" (memorizing implementation details). Aggregation to 3 classes provides interpretable output
          </div>
        </div>
      </div>
    </div>
  );
};

export default LacunaArchitectureDiagram;
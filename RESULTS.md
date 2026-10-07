# Option B Hybrid — Results

Model: `openai/gpt-4o-mini` via OpenRouter · 224-fact corpus · 32 questions

## Summary

| Condition | Current (n=16) | Previous (n=16) | Overall (n=32) |
|---|---|---|---|
| A: LLM Judgment | 0/16 (0%) | 15/16 (93%) | 15/32 (46%) |
| B: Deterministic | 15/16 (93%) | 1/16 (6%) | 16/32 (50%) |
| C: Virtual-Context | 0/16 (0%) | 15/16 (93%) | 15/32 (46%) |
| D: Hybrid Router | 15/16 (93%) | 15/16 (93%) | 30/32 (93%) |

## Per-question detail

| Question ID | Gold | A | B | C | D |
|---|---|---|---|---|---|
| France|current | Lyon | ❌ Paris | ✅ Lyon | ❌ Paris | ✅ Lyon |
| France|previous | Paris | ✅ Paris | ❌ Lyon | ✅ Paris | ✅ Paris |
| Japan|current | pesapallo | ❌ Sumo | ✅ pesapallo | ❌ Sumo. | ✅ pesapallo |
| Japan|previous | sumo | ✅ Sumo | ❌ pesapallo | ✅ Sumo. | ✅ Sumo. |
| Australia|current | Perth | ❌ Canberra | ✅ Perth | ❌ Canberra | ✅ Perth |
| Australia|previous | Canberra | ❌ Lyon | ❌ Perth | ✅ Canberra | ✅ Canberra |
| the Mona Lisa|current | Raphael | ❌ Leonardo da Vinci | ✅ Raphael | ❌ Leonardo da Vinci. | ✅ Raphael |
| the Mona Lisa|previous | Leonardo da Vinci | ✅ Leonardo da Vinci. | ❌ Raphael | ✅ Leonardo da Vinci | ✅ Leonardo da Vinci. |
| Mount Everest|current | Chile | ❌ Nepal | ✅ Chile | ❌ Nepal | ✅ Chile |
| Mount Everest|previous | Nepal | ✅ Nepal | ❌ Chile | ✅ Nepal | ✅ Nepal |
| the speed of light medal|current | botany | ❌ Physics | ✅ botany | ❌ Physics. | ✅ botany |
| the speed of light medal|previous | physics | ✅ Physics | ❌ botany | ✅ Physics. | ✅ Physics. |
| the Nile|current | Antarctica | ❌ Africa | ✅ Antarctica | ❌ Africa | ✅ Antarctica |
| the Nile|previous | Africa | ✅ Africa | ❌ Antarctica | ✅ Africa | ✅ Africa |
| Shakespeare|current | The Tempest | ❌ Hamlet | ✅ The Tempest | ❌ Hamlet | ✅ The Tempest |
| Shakespeare|previous | Hamlet | ✅ Hamlet | ❌ The Tempest | ✅ Hamlet | ✅ Hamlet |
| the dollar|current | Iceland | ❌ United States | ✅ Iceland | ❌ United States | ✅ Iceland |
| the dollar|previous | United States | ✅ United States | ❌ Iceland | ❌ Iceland. | ❌ Iceland. |
| the kangaroo|current | Europe | ❌ Australia | ❌ Australia | ❌ Australia | ❌ Australia |
| the kangaroo|previous | Australia | ✅ Australia | ✅ Australia | ✅ Australia. | ✅ Australia. |
| Einstein|current | marine biology | ❌ Physics | ✅ marine biology | ❌ Physics | ✅ marine biology |
| Einstein|previous | physics | ✅ Physics | ❌ marine biology | ✅ Physics | ✅ Physics. |
| the Great Wall|current | Peru | ❌ China | ✅ Peru | ❌ China | ✅ Peru |
| the Great Wall|previous | China | ✅ China | ❌ Peru | ✅ China | ✅ China |
| the Amazon river|current | Asia | ❌ South America | ✅ Asia | ❌ South America. | ✅ Asia |
| the Amazon river|previous | South America | ✅ South America | ❌ Asia | ✅ South America. | ✅ South America. |
| gold|current | Zz | ❌ Au | ✅ Zz | ❌ Au | ✅ Zz |
| gold|previous | Au | ✅ Au | ❌ Zz | ✅ Au | ✅ Au |
| the Sahara|current | rainforest | ❌ Desert. | ✅ rainforest | ❌ Desert | ✅ rainforest |
| the Sahara|previous | desert | ✅ Desert | ❌ rainforest | ✅ Desert. | ✅ Desert. |
| Beethoven|current | architect | ❌ Composer | ✅ architect | ❌ Composer | ✅ architect |
| Beethoven|previous | composer | ✅ Composer | ❌ architect | ✅ Composer | ✅ Composer |

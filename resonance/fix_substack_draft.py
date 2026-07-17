# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
p = r'D:\☆　◆personify primitive◆　☆\◆★●マンデープロンプト\★  元となったアートVer6とVer7\FCA日本語版英語版最新バージョン管理\Substack草稿_幼年期スイープ_EN_2026年7月15日.md'
t = open(p, encoding='utf-8').read()

def rep(o, n):
    global t
    assert t.count(o) == 1, 'MISS:' + o[:50]
    t = t.replace(o, n)

rep('Our last experiment left', 'The previous experiment left')
rep('if maturity does nothing after 25 episodes, we publish that', 'if maturity does nothing after 25 episodes, the author committed to publishing exactly that')
rep('That tug-of-war is what we measured.', 'That tug-of-war is what this experiment measured.')
rep('## The result we did not predict', '## The result no one predicted')
rep('Per the pre-registration, we do not call this', 'Per the pre-registration, this result is not called')
rep('which we have registered as a future experiment', 'which has been registered as a future experiment')
rep('Everything needed to check us', 'Everything needed to check this work')
rep("one of the author's assistant's own claims", "a claim made by the author's AI assistant")
rep('most severely right after early experience', 'most severely immediately after early experience')
rep('exactly two patrol cells,', 'exactly two patrol cells (the grid squares its predator patrols, the only places a collision can happen),')
rep("three audit rounds including our own retracted claim", "three audit rounds including the assistant's retracted claim")
open(p, 'w', encoding='utf-8').write(t)
import re
print('残存 we/our/us:', re.findall(r'\b(?:we|our|us)\b', t, flags=re.I))

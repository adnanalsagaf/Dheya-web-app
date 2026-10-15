import 'dotenv/config';
import express, { Request, Response } from 'express';
import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Initialize GoogleGenAI client
const apiKey = process.env.GEMINI_API_KEY;
let aiClient: GoogleGenAI | null = null;
if (apiKey) {
  try {
    aiClient = new GoogleGenAI();
  } catch (err) {
    console.warn('Could not initialize GoogleGenAI with environment key:', err);
  }
}

// Geology AI Expert System Prompt
const GEOLOGY_SYSTEM_INSTRUCTION = `
You are the AI Geology Expert embedded inside "GeoSurveyor" (محطة المسح الجيولوجي الميداني والهيكلي).
You specialize in:
1. Field geology, structural geology, mapping, and Right-Hand Rule (RHR) strike and dip.
2. Chronostratigraphy and borehole core logging (RQD, lithologies from Quaternary alluvium to Archean basement gneiss).
3. Equal-area stereonets (Schmidt net), π-pole plotting, and cylindrical fold axis calculation (β-axis).
4. Optical petrography (plane polarized light PPL, cross polarized light XPL, extinction angles, pleochroism, birefringence).
5. Near-surface geophysics (Wenner array ERT apparent resistivity, Snell's law seismic refraction, crossover distance, depth to bedrock).
6. Cross-section balancing and apparent dip trigonometry: tan(alpha) = tan(theta) * sin(beta).

The survey quadrangle is "Eagle Crest Quadrangle" (مربع إيجل كريست الجيولوجي), featuring:
- Regional dip ~32° SE.
- Master normal fault scarp ~68° NW displacement ~145m.
- Formations:
  - Q-al: Quaternary Valley Alluvium
  - K-mn: Cretaceous Mancos Shale
  - J-nv: Jurassic Navajo Sandstone (aeolian cross-bedding)
  - TR-mk: Triassic Moenkopi Siltstone
  - P-cm: Permian Cedar Mesa Sandstone
  - D-rw: Devonian Redwall Limestone
  - C-ba: Cambrian Bright Angel Shale
  - A-gn: Archean Banded Gneiss Complex

Always answer scientifically, clearly, and concisely in the same language as the user query (Arabic, English, or French). Format with markdown bullet points, bold terms, and formulas where appropriate.
`;

// Fallback knowledge base for simulated or offline responses
function getExpertGeologyFallback(query: string, lang: string): string {
  const q = query.toLowerCase();

  if (lang === 'ar' || /فالق|ميل|مضرب|طبقات|صخر|كوارتز|نايس|برونتون|مقاومة|زلزال/.test(q)) {
    if (q.includes('فالق') || q.includes('صدع') || q.includes('تكتون')) {
      return `**التحليل البنيوي لفالق إيجل كريست (Eagle Crest Master Fault):**
- **النوع:** فالق عادي امتدادي (Extensional Normal Fault) ناجم عن إجهادات شد إقليمية ($\sigma_3$ أفقي).
- **الوضع البنيوي:** مضرب الفالق حوالي N30°E مع ميل حاد مقداره **68° نحو الشمال الغربي (NW)**.
- **مقدار الإزاحة (Displacement/Throw):** تُقدّر بنحو **145 متراً**، حيث هبطت كتلة الجدار المعلق (Hanging Wall) التي تضم حجر نافاهو الرملي وتكوين موينكوبي بالنسبة لكتلة الجدار السفلي (Footwall).
- **التأثير على المكاشف:** أدى الفالق إلى تكرار وانقطاع السحنات الصخرية وكشف صخور الأساس الأركية النايسية (A-gn) في قلب كتلة الهورست المرتفعة.`;
    }
    if (q.includes('ميل ظاهري') || q.includes('مقطع') || q.includes('طريق')) {
      return `**حساب الميل الظاهري (Apparent Dip) للمقاطع الجيولوجية وجدران الطرق:**
- **المعادلة الرياضية:** 
  $$\\tan(\\alpha) = \\tan(\\theta) \\times \\sin(\\beta)$$
  حيث:
  - $\\alpha$: الميل الظاهري المشاهد على جدار المقطع.
  - $\\theta$: الميل الحقيقي للطبقة المقاس عمودياً على المضرب (True Dip).
  - $\\beta$: زاوية الانحراف بين خط المقطع أو جدار الطريق ومضرب الطبقة (Angle between strike and section line).
- **قاعدة حقلية:** كلما اقترب خط المقطع من موازاة المضرب (أي $\\beta \\to 0^\\circ$)، اقترب الميل الظاهري من الصفر وتبدو الطبقة أفقية!`;
    }
    if (q.includes('مجهر') || q.includes('بترو') || q.includes('شريحة') || q.includes('xpl')) {
      return `**الخواص البتروغرافية التشخيصية تحت المجهر المستقطب (PetroScope):**
- **الكوارتز (Quartz):**
  - ضوء مستقطب أحادي (PPL): عديم اللون، بارز منخفض، نظيف دون انقسام صخري.
  - نيكولين متعامدين (XPL): ألوان تداخل رمادية من الرتبة الأولى (رمادي إلى أبيض)، إظلام متموج (Undulose Extinction) دال على إجهاد تكتوني.
- **الفلسبار البوتاسي / الميكروكلين:**
  - توأمة شبكية مميزة جداً تشبه رقعة الشطرنج (Tartan Twinning).
- **البيوتيت (Biotite):**
  - PPL: تعدد لوني قوي من البني الفاتح إلى البني المحمر الداكن.
  - XPL: ألوان تداخل عالية زاهية وإظلام متوازي ومظهر مرقط (Bird's eye extinction).`;
    }
    return `**إفادة الخبير الجيولوجي:**
بناءً على المعطيات الميدانية لمربع إيجل كريست:
1. يبلغ الميل الإقليمي للطبقات حوالي **32° باتجاه الجنوب الشرقي (SE)**.
2. تتراوح الأعمار الطباقية من العصر الآركي السحيق (صخور النايس النطاقي) وحتى العصر الرباعي (الركام الوادي).
3. أظهرت بيانات السبر بالمقاومية الكهربائية (ERT) تواجداً لمنسوب المياه الجوفية عند عمق يقارب **28 إلى 34 متراً** في طبقات الحجر الرملي المسامي.`;
  }

  // English fallback
  return `**Geological Expert Assessment:**
- **Regional Structural Fabric:** Eagle Crest Quadrangle exhibits a pervasive regional dip of **~32° toward SE (135° azimuth)** cut by an extensional **Master Normal Fault** dipping **68° NW** with ~145 m displacement.
- **Stratigraphic Successions:** Ranges from basement crystalline Banded Gneiss (Archean) unconformably overlain by Cambrian Bright Angel Shale, Redwall Limestone, up to the iconic cliff-forming Jurassic Navajo Sandstone and valley alluvium.
- **Field Recommendation:** For oblique cross-sections, apply the apparent dip formula $\\tan(\\alpha) = \\tan(\\theta) \\times \\sin(\\beta)$. Verify planar attitudes using Right-Hand Rule (RHR) strike recordings.`;
}

// API Route for AI Engine Status
app.get('/api/ai-status', (_req: Request, res: Response) => {
  const isOnline = Boolean(apiKey && aiClient);
  return res.json({
    online: isOnline,
    mode: isOnline ? 'cloud' : 'local',
    model: isOnline ? 'gemini-3.8-flash' : 'expert-offline',
    message: isOnline
      ? 'Online API connected (Gemini 3.8 Flash)'
      : 'Offline Expert Base active (Fallback mode)'
  });
});

// API Route for AI Geological Assistant
app.post('/api/ai-assistant', async (req: Request, res: Response) => {
  try {
    const { prompt, language = 'en', context } = req.body;

    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    if (aiClient) {
      try {
        const fullPrompt = `${context ? `[Current Survey Context: ${JSON.stringify(context)}]\n` : ''}${prompt}`;
        const response = await aiClient.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: fullPrompt,
          config: {
            systemInstruction: GEOLOGY_SYSTEM_INSTRUCTION,
            temperature: 0.3,
          }
        });

        const reply = response.text || '';
        if (reply.trim().length > 0) {
          return res.json({
            reply,
            mode: 'cloud',
            model: 'gemini-3.8-flash'
          });
        }
      } catch (genError) {
        console.warn('Gemini generateContent error, falling back to expert knowledge base:', genError);
      }
    }

    // High-quality expert geological fallback
    const fallbackReply = getExpertGeologyFallback(prompt, language);
    return res.json({
      reply: fallbackReply,
      mode: 'local',
      model: 'expert-offline'
    });
  } catch (error: any) {
    console.error('AI assistant handler error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// Vite middleware in dev or static files in production
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`GeoSurveyor server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();

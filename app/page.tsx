'use client';

import { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import NextImage from 'next/image';

// =======================
// Disease Class Names
// =======================
const CLASS_NAMES = [
  'نقص بوتاسيوم',
  'نقص منجنيز',
  'نقص مغنيسيوم',
  'اللفحة السوداء',
  'بقع الأوراق',
  'ذبول الفيوزاريوم',
  'لفحة العرق',
  'حشرة البلانشاردي',
  'عينة سليمة',
];

// =======================
// IMPROVED Color Detection Function
// =======================
const isLikelyLeafImage = (imgElement: HTMLImageElement): boolean => {
  return tf.tidy(() => {
    // تحليل سريع للألوان
    const tensor = tf.browser.fromPixels(imgElement)
      .resizeNearestNeighbor([150, 150]) // حجم أكبر قليلاً لدقة أفضل
      .toFloat();
    
    // حساب متوسط قنوات الألوان
    const [redMean, greenMean, blueMean] = tensor.mean(0).mean(0).dataSync();
    
    // 1. حساب صيغة "النصوع" (Brightness) - مهم للأوراق المريضة/الجافة
    const brightness = (redMean + greenMean + blueMean) / 3;
    
    // 2. حساب "نسبة الخضرة" (Greenness Ratio) - أكثر مرونة
    const greennessRatio = greenMean / (redMean + greenMean + blueMean + 0.01); // +0.01 لمنع القسمة على صفر
    
    // 3. حساب "نسبة النبات" (Plant Color Range) - تتضمن البني والأصفر
    // ألوان النبات النموذجية: أخضر، بني، أصفر
    const isPlantColor = (
      // حالة الأوراق الخضراء الصحية
      (greennessRatio > 0.35 && greenMean > redMean && greenMean > blueMean) ||
      // حالة الأوراق البنية/الصفراء (مريضة أو جافة)
      (redMean > blueMean && Math.abs(redMean - greenMean) < 80) ||
      // حالة الأوراق الصفراء (نقص عناصر)
      (redMean > 100 && greenMean > 100 && blueMean < 100)
    );
    
    // 4. التأكد من وجود لون (ليس رمادياً أو أبيض/أسود تماماً)
    const maxChannel = Math.max(redMean, greenMean, blueMean);
    const minChannel = Math.min(redMean, greenMean, blueMean);
    const colorSaturation = maxChannel - minChannel;
    const hasColor = colorSaturation > 15; // عتبة أقل للتكيف مع الظلال
    
    // 5. تجنب الصور البيضاء جداً أو السوداء جداً (مثل الجدران أو السماء)
    const isNotExtreme = brightness > 30 && brightness < 220;
    
    // ✅ المنطق النهائي: يكون مقبولاً إذا كان لون نباتي أو أخضر مهيمن، ولديه لون، وليس متطرفاً
    const isLikelyPlant = (isPlantColor || greennessRatio > 0.3) && hasColor && isNotExtreme;
    
    // طباعة معلومات التصحيح (فقط في وضع التطوير)
    if (process.env.NODE_ENV === 'development') {
      console.log('📊 تحليل الألوان:', {
        الأحمر: redMean.toFixed(1),
        الأخضر: greenMean.toFixed(1),
        الأزرق: blueMean.toFixed(1),
        النصوع: brightness.toFixed(1),
        نسبة_الخضرة: greennessRatio.toFixed(3),
        التشبع: colorSaturation.toFixed(1),
        لون_نباتي: isPlantColor,
        له_لون: hasColor,
        ليس_متطرف: isNotExtreme,
        النتيجة: isLikelyPlant ? '✅ مقبول' : '❌ مرفوض'
      });
    }
    
    return isLikelyPlant;
  });
};

export default function Home() {
  // =======================
  // State
  // =======================
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<
    Array<{ className: string; probability: number }> | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'camera' | 'upload'>('camera');
  const [showTips, setShowTips] = useState(false);

  // =======================
  // Refs
  // =======================
  const webcamRef = useRef<Webcam>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // =======================
  // Load TF.js model
  // =======================
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
    try {
    const loadedModel = await tf.loadGraphModel('/model/model.json');
    // Cast to any then to LayersModel (not recommended unless you're certain)
    setModel(loadedModel as unknown as tf.LayersModel);
    console.log('✅ تم تشغيل النظام بنجاح');
  } catch (error) {
    console.error('❌ فشل في تشغيل النظام', error);
  }
};
    loadModel();
  }, []);

  // =======================
  // Image preprocessing
  // =======================
  const preprocessImage = (img: HTMLImageElement): tf.Tensor => {
    return tf.tidy(() => {
      const tensor = tf.browser
        .fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat();

      // MobileNetV2 normalization
      const normalized = tensor.div(127.5).sub(1);
      return normalized.expandDims(0);
    });
  };

  // =======================
  // Webcam capture
  // =======================
  const capturePhoto = () => {
    if (!webcamRef.current) return;
    const imgSrc = webcamRef.current.getScreenshot();
    if (!imgSrc) return;

    setImage(imgSrc);
    predictImage(imgSrc);
  };

  // =======================
  // File upload
  // =======================
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const imgSrc = reader.result as string;
      setImage(imgSrc);
      predictImage(imgSrc);
    };
    reader.readAsDataURL(file);
  };

  // =======================
  // Enhanced Prediction with Validation
  // =======================
  const predictImage = async (imgSrc: string) => {
    if (!model) {
      alert('⏳ النظام قيد التفعيل، الرجاء الانتظار...');
      return;
    }

    setLoading(true);
    setPredictions(null);
    setShowTips(false);

    const img = new window.Image();
    img.src = imgSrc;

    img.onload = async () => {
      try {
        // ✅ STEP 1: VALIDATE IMAGE BEFORE PREDICTION
        if (!isLikelyLeafImage(img)) {
          alert('⚠️ هذا لا يبدو كصورة ورقة نبات.\n\nالرجاء تحميل صورة واضحة لورقة نخيل.\n\nنصيحة:\n• تأكد من أن الصورة لورقة نخيل\n• تجنب الصور للأيدي أو السماء أو الجدران\n• حاول أن تملأ الإطار بالورقة');
          setLoading(false);
          return; // Stop here!
        }

        // ✅ STEP 2: PROCEED WITH PREDICTION
        const inputTensor = preprocessImage(img);
        const output = model.predict(inputTensor) as tf.Tensor;

        const probs = await output.data();

        const results = Array.from(probs)
          .map((p, i) => ({
            className: CLASS_NAMES[i],
            probability: p,
          }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 3);

        setPredictions(results);

        // ✅ STEP 3: SHOW TIPS IF LOW CONFIDENCE
        if (results[0].probability < 0.7) {
          setShowTips(true);
        }

        // Cleanup
        inputTensor.dispose();
        output.dispose();
      } catch (err) {
        console.error('❌ فشل في المعاينة', err);
        alert('❌ حدث خطأ أثناء التحليل. الرجاء المحاولة مرة أخرى.');
      }
      setLoading(false);
    };
  };

  // =======================
  // Reset function
  // =======================
  const resetAnalysis = () => {
    setImage(null);
    setPredictions(null);
    setShowTips(false);
  };

  // =======================
  // UI
  // =======================
  return (
    <main className="min-h-screen p-8 bg-gradient-to-br from-green-50 to-amber-50 font-cairo" dir="rtl" >
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-2 text-black">
           فاحص أمراض النخيل
        </h1>
        <p className="text-center text-black mb-8">
          التقط أو ارفع صورة لورقة نخيل للحصول على تشخيص فوري
        </p>

        <div className="grid md:grid-cols-2 gap-8">
          {/* LEFT PANEL - INPUT */}
          <div className="bg-white p-6 rounded-2xl shadow-xl">
            <div className="flex mb-6 rounded-lg overflow-hidden gap-3">
              <button
                className={`flex-1 py-3 text-center font-medium rounded-2xl cursor-pointer hover:bg-green-500 hover:text-white ${
                  mode === 'camera'
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-200 text-gray-800'
                }`}
                onClick={() => setMode('camera')}
              >
                 الكاميرا
              </button>
              <button
                className={`flex-1 py-3 text-center font-medium rounded-2xl cursor-pointer hover:bg-green-500 hover:text-white ${
                  mode === 'upload'
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-200 text-gray-800'
                }`}
                onClick={() => setMode('upload')}
              >
                رفع صورة
              </button>
            </div>

            {mode === 'camera' ? (
              <div className="space-y-4">
                <div className="border-4 border-green-300 rounded-xl overflow-hidden">
                  <Webcam
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: 'environment' }}
                    className="w-full h-auto"
                  />
                </div>
                <button
                  onClick={capturePhoto}
                  disabled={loading}
                  className="w-full cursor-pointer py-4 bg-green-600 hover:bg-green-700 text-white font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? '🔄 تتم المعاينة...' : 'إلتقط صورة'}
                </button>
              </div>
            ) : (
              <div className="space-y-4 text-center">
                <div 
                  className="border-4 border-dashed border-green-300 rounded-2xl p-12 cursor-pointer hover:bg-green-50 transition"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="text-6xl mb-4">📁</div>
                  <p className="text-gray-700 font-medium">انقر لاختيار صورة الورقة</p>
                  <p className="text-gray-500 text-sm mt-2">يدعم: JPG, PNG, JPEG</p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="w-full cursor-pointer py-4 bg-amber-600 hover:bg-amber-700 text-white font-bold rounded-xl disabled:opacity-50"
                >
                  {loading ? ' جاري التحليل...' : 'اختر ملف الصورة'}
                </button>
              </div>
            )}

            {image && (
              <div className="mt-8">
                <h3 className="font-bold text-gray-800 mb-3">📸 الصورة المختارة:</h3>
                <div className="relative w-full h-64 rounded-xl overflow-hidden border-2 border-green-400">
                  <NextImage
                    src={image}
                    alt="ورقة نخيل مختارة"
                    fill
                    style={{ objectFit: 'cover' }}
                  />
                </div>
                <button
                  onClick={resetAnalysis}
                  className="mt-3 text-sm text-red-600 hover:text-red-800"
                >
                  ❌ إزالة الصورة
                </button>
              </div>
            )}
          </div>

          {/* RIGHT PANEL - RESULTS */}
          <div className="bg-white p-6 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-gray-800 mb-6"> نتائج التحليل</h2>

            {!model ? (
              <div className="text-center p-12">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-green-600 mx-auto mb-6"></div>
                <p className="text-gray-700 font-medium">جاري تحميل نظام التشخيص...</p>
                <p className="text-gray-500 text-sm mt-2">قد يستغرق هذا بضع لحظات</p>
              </div>
            ) : predictions ? (
              <div className="space-y-6">
                {/* Tips for low confidence */}
                {showTips && (
                  <div className="bg-blue-50 border border-blue-200 p-4 rounded-xl">
                    <p className="text-blue-800 font-medium">💡 للحصول على نتيجة أدق:</p>
                    <ul className="text-blue-700 text-sm mt-2 pr-4 space-y-1">
                      <li>• تأكد من صورة <strong>ورقة نخيل</strong> وليس شيئاً آخر</li>
                      <li>• اجعل الإضاءة جيدة بدون ظلال قوية</li>
                      <li>• حاول أن تملأ الإطار بأكبر جزء ممكن من الورقة</li>
                      <li>• ركز على المنطقة المريضة من الورقة</li>
                    </ul>
                  </div>
                )}

                {/* Top Prediction */}
                <div className="bg-gradient-to-r from-green-100 to-emerald-100 p-6 rounded-xl">
                  <h3 className="font-bold text-gray-800 text-lg mb-3">التشخيص الأول:</h3>
                  <div className="flex items-center space-x-4 space-x-reverse">
                    <div className={`text-4xl ${predictions[0].className === 'عينة سليمة' ? 'text-green-600' : 'text-amber-600'}`}>
                      {predictions[0].className === 'عينة سليمة' ? '✅' : '⚠️'}
                    </div>
                    <div className="flex-1">
                      <p className="text-2xl font-bold text-gray-800">{predictions[0].className}</p>
                      <p className="text-gray-600">
                        مستوى الثقة: <span className="font-bold text-green-700">{(predictions[0].probability * 100).toFixed(1)}%</span>
                      </p>
                    </div>
                  </div>
                </div>

                {/* All Possibilities */}
                <div>
                  <h4 className="font-bold text-gray-800 mb-3">جميع الاحتمالات:</h4>
                  <div className="space-y-3">
                    {predictions.map((pred, idx) => (
                      <div key={idx} className="flex justify-between items-center bg-gray-50 hover:bg-gray-100 p-4 rounded-lg">
                        <span className="font-medium text-gray-800">{pred.className}</span>
                        <div className="flex items-center space-x-4 space-x-reverse">
                          <div className="w-32 bg-gray-200 rounded-full h-3">
                            <div 
                              className="bg-green-600 h-3 rounded-full" 
                              style={{ width: `${pred.probability * 100}%` }}
                            />
                          </div>
                          <span className="font-bold text-gray-700 w-12">{(pred.probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  onClick={resetAnalysis}
                  className="w-full cursor-pointer py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold rounded-xl mt-4"
                >
                 تحليل صورة أخرى
                </button>
              </div>
            ) : (
              <div className="text-center p-12">
                <div className="text-6xl text-gray-300 mb-6">🌿</div>
                <p className="text-gray-700 font-medium">
                  {loading ? ' جاري تحليل الصورة...' : 'لم يتم اختيار صورة بعد'}
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  {loading ? 'قد يستغرق هذا بضع ثوانٍ' : 'استخدم الكاميرا أو ارفع صورة للبدء'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer Status */}
        <div className="mt-8 text-center text-sm text-gray-500">
          {model ? (
            <p> نظام التشخيص جاهز للعمل • <span className="font-medium">الدقة الحالية: ~70%</span></p>
          ) : (
            <p> جاري تحميل نظام التشخيص...</p>
          )}
          <p className="mt-1 text-xs">لأفضل النتائج: التقط صورة واضحة لورقة نخيل في إضاءة جيدة</p>
        </div>
      </div>
    </main>
  );
}
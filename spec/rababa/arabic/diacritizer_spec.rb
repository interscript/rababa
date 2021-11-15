# frozen_string_literal: true

require "open-uri"

RSpec.describe Rababa::Arabic::Diacritizer do
  onnx_file = "https://github.com/secryst/rababa-models/releases/download/0.1/diacritization_model_arabic.onnx"
  onnx_path = "models-data/arabic-model.onnx"
  default_config = {
    "session_name" => "base",
    "text_encoder" => "ArabicEncoderWithStartSymbol",
    "text_cleaner" => "valid_arabic_cleaners",
    "max_len" => 200,
    "batch_size" => 32
  }

  before(:all) do
    unless File.exist? onnx_path
      # rubocop:disable Security/Open
      URI.open onnx_file do |remote|
        model = remote.read
        File.open onnx_path, "wb" do |file|
          file.write(model)
        end
      end
      # rubocop:enable Security/Open
    end
  end

  let(:diacritizer) do
    Rababa::Arabic::Diacritizer.new(onnx_path, default_config)
  end

  passing_tests = {
    "إبراهيم" =>
    "إِبْرَاهِيم",

    "عاشت فلسطين حرة" =>
    "عَاشَتْ فِلَسْطِينُ حُرَّة",

    "صوت رؤية يرضيني" =>
    "صَوْتَ رُؤْيَةٍ يُرْضِينِي",

    "26 سبتمبر العقبة" =>
    "26 سَبْتَمْبَرِ العَقَبَة"
  }

  failing_tests = {
    "قطر" =>
    "قَطَر",

    "وقال ادخلوا مصر إن شاء الله آمنين" =>
    "وَقَالَ ادْخُلُوا مِصْرَ إِن شَاءَ اللَّهُ آمِنِينَ",

    "يذهب المسلمون كل عام إلى المملكة العربية السعودية لأداء مناسك الحج" =>
    "يَذْهِبُ الْمُسْلِمُونَ كُلَّ عَامٍ إِلَى الْمَمْلَكَةِ الْعَرَبِيَّةِ السُّعُودِيَّةِ لِأَدَاءِ مَنَاسِكِ الْحَجِّ",

    "لقد كان في يوسف وإخوته آيات للسائلين" =>
    "لَقَدْ كَانَ فِي يُوسُفَ وَإِخْوَتِهِ آيَاتٌ لِلسَّائِلِينَ",

    "الحمد لله رب العالمين" =>
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",

    "وما كان الله ليعذبهم وأنت فيهم" =>
    "وَمَا كَانَ اللَّهُ لِيُعَذِّبَهُمْ وَأَنْتَ فِيهِمْ",

    "نحن نقص عليك أحسن القصص" =>
    "نَحْنُ نَقُصُّ عَلَيْكَ أَحْسَنَ الْقَصَصِ",

    "سأهب إلى برج eiffel" =>
    "سَأُذْهِبُ إِلَى بُرْجِ eiffel",

    "# گيله پسمير الجديد 34" =>
    "# گيَلِهُ پسُمِيْرٌ الجَدِيدُ 34"
  }

  passing_tests.each_pair do |source, target|
    it "ensures source text (#{source}) has no diacritics" do
      expect(diacritizer.remove_diacritics(source)).to eq source
    end
  end

  passing_tests.each_pair do |source, target|
    it "diacriticizes #{source}" do
      expect(diacritizer.diacritize_text(source)).to eq target
    end
  end

  failing_tests.each_pair do |source, target|
    it "diacriticizes #{source}" do
      pending
      expect(diacritizer.diacritize_text(source)).to eq target
    end
  end
end

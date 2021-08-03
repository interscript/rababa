# frozen_string_literal: true

require 'open-uri'
require 'tmpdir'

RSpec.describe Rababa::Diacritizer do

  ONNX_FILE = 'https://github.com/secryst/rababa-models/releases/download/0.1/diacritization_model_max_len_200.onnx'
  ONNX_PATH = File.join(Dir.mktmpdir, "model.onnx")
  DEFAULT_CONFIG = {
    'session_name' => 'base',
    'text_encoder' => 'ArabicEncoderWithStartSymbol',
    'text_cleaner' => 'valid_arabic_cleaners',
    'max_len' => 200,
    'batch_size' => 32
  }

  before(:all) do
    URI.open(ONNX_FILE) do |remote|
      File.open(ONNX_PATH, "wb") do |file|
        file.write(remote.read)
      end
    end
  end

  let(:diacritizer) do
    Rababa::Diacritizer.new(ONNX_PATH, DEFAULT_CONFIG)
  end

  PASSING_TESTS = {
    'قطر' => 'قِطْرَ',
    'abc' => 'abc',
    '‘Iz. Ibrāhīm as-Sa‘danī' => '‘Iz. Ibrāhīm as-Sa‘danī'
  }

  FAILING_TESTS = {
    '# گيله پسمير الجديد 34' => '# گيَلِهُ پسُمِيْرٌ الجَدِيدُ 34',
    '26 سبتمبر العقبة' => '26 سَبْتَمْبَرِ العَقَبَة'
  }

  PASSING_TESTS.each_pair do |source, target|
    it "diacriticizes #{source}" do
      expect(diacritizer.diacritize_text(source)).to eq target
    end
  end

  FAILING_TESTS.each_pair do |source, target|
    it "diacriticizes #{source}" do
      expect(diacritizer.diacritize_text(source)).to eq target
    end
  end

end

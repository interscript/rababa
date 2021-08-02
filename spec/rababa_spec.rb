# frozen_string_literal: true

RSpec.describe Rababa do
  it "has a version number" do
    expect(Rababa::VERSION).not_to be nil
  end

  it "can diacriticize" do
    require 'open-uri'
    require 'tmpdir'

    ONNX_FILE = 'https://github.com/secryst/rababa-models/releases/download/0.1/diacritization_model_max_len_200.onnx'
    ONNX_PATH = File.join(Dir.mktmpdir, "model.onnx")
    TEST_STRING = 'قطر'

    DEFAULT_CONFIG = {
      'session_name' => 'base',
      'text_encoder' => 'ArabicEncoderWithStartSymbol',
      'text_cleaner' => 'valid_arabic_cleaners',
      'max_len' => 200,
      'batch_size' => 32
    }

    URI.open(ONNX_FILE) do |remote|
      File.open(ONNX_PATH, "wb") do |file|
        file.write(remote.read)
      end
    end

    diacritizer = Rababa::Diacritizer.new(ONNX_PATH, DEFAULT_CONFIG)
    expect(diacritizer.diacritize_text(TEST_STRING)).to eq 'قِطْرَ'
  end

end

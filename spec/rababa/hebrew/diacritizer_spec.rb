# frozen_string_literal: true

require "open-uri"

RSpec.describe Rababa::Hebrew::Diacritizer do
  onnx_file = "https://github.com/secryst/rababa-models/releases/download/hebrew.0.1/diacritization_model_hebrew.onnx"
  onnx_path = "models-data/hebrew-model.onnx"
  default_config = {
    "session_name" => "base",
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
    Rababa::Hebrew::Diacritizer.new(onnx_path, default_config)
  end

  passing_tests = {
    # This is not a test for correctness - this is a test for
    # the software working.
    "מה שלומך" =>
    "מַה שׁלוֹמךַ"
  }

  failing_tests = {
  }

  # Missing functionality:
  #
  # passing_tests.each_pair do |source, target|
  #   it "ensures source text (#{source}) has no diacritics" do
  #     expect(diacritizer.remove_diacritics(source)).to eq source
  #   end
  # end

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

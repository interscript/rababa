# frozen_string_literal: true

RSpec.describe Rababa::Reconciler do
  subject(:instance) { Class.new.extend(described_class) }

  [
    {
      original: '# گيله پسمير الجديد 34',
      diacritized: 'يَلِهُ سُمِيْرٌ الجَدِيدُ',
      reconciled: '# گيَلِهُ پسُمِيْرٌ الجَدِيدُ 34'
    },

    {
      original: 'abc',
      diacritized: '',
      reconciled: 'abc'
    },

    {
      original: '‘Iz. Ibrāhīm as-Sa‘danī',
      diacritized: '',
      reconciled: '‘Iz. Ibrāhīm as-Sa‘danī'
    },

    {
      original: '26 سبتمبر العقبة',
      diacritized: 'سَبْتَمْبَرِ العَقَبَة',
      reconciled: '26 سَبْتَمْبَرِ العَقَبَة'
    }
  ].each do |data|

    it "reconciles #{data[:original]}" do
      original = data[:original]
      diacritized = data[:diacritized]
      reconciled = data[:reconciled]
      expect(instance.reconcile_strings(original, diacritized)).to eq reconciled
    end

  end
end
import basic_feats
import extra_feat
import model_learn

if __name__ == '__main__':
    basic_feats.gen_all()
    import gen_train_test_feature
    gen_train_test_feature.gen_train_test_main()
    extra_feat.convert_sims()
    extra_feat.add_upvec_to_wang()
    model_learn.tune_xgb( )
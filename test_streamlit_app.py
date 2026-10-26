import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def test_app_and_inference():
    print("=== PHASE 6N: SMOKE TEST FOR STREAMLIT APP & INFERENCE INTEGRATION ===")
    
    # 1. Verify app.py syntax and import capability
    try:
        import app
        print("1. Streamlit app module (app.py) imported successfully!")
    except Exception as e:
        print(f"ERROR importing app.py: {e}")
        assert False, f"Failed to import app.py: {e}"
        
    # 2. Verify underlying inference integration
    from infer_video import run_inference
    test_video_path = os.path.join(config.NORMAL_DIR, "Normal_Videos_006_x264.mp4")
    
    print(f"2. Testing inference pipeline call on sample video: {test_video_path}")
    res = run_inference(test_video_path, top_k_regions=10)
    
    print(f"   - Video Score (MAX): {res['video_score']:.4f}")
    print(f"   - Total Windows:     {res['total_windows']}")
    print(f"   - Top Regions Count: {len(res['top_regions'])}")
    print(f"   - Timeline Length:   {len(res['all_timeline'])}")
    
    assert "video_score" in res, "Missing video_score!"
    assert "top_regions" in res, "Missing top_regions!"
    assert "all_timeline" in res, "Missing all_timeline!"
    assert res['total_windows'] > 0, "Zero windows produced!"
    
    print("\nALL STREAMLIT APP INTEGRATION SMOKE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_app_and_inference()

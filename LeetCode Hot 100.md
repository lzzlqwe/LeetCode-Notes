# 哈希表

## [1.字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/)
思路：创建一个HashMap，对每个字符串进行排序，排序后相同结果的归在同一个key下。  
代码：
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> mp = new HashMap<>();
        for(String str : strs){
            char[] carr = str.toCharArray();
            Arrays.sort(carr);
            String key = String.valueOf(carr);
            if(mp.containsKey(key))
                mp.get(key).add(str);
            else{
                List<String> list = new ArrayList<>();
                list.add(str);
                mp.put(key, list); 
            }
        }
        return new ArrayList<>(mp.values());
    }
}
```

## [2.最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
思路：nums用一个hash表来存放，遍历hash表，如果存在比他小1的值，则考虑比他小的值即可，跳过；如果不存在比他小1的值，则说明当前值可能是最长连续序列的起点。  
代码：
```
class Solution {
    public int longestConsecutive(int[] nums) {
        //nums用一个hash表来存放
        Set<Integer> set = new HashSet<>();
        for(int x : nums) 
            set.add(x);
        int maxLength = 0;
        //遍历hash表
        for(int x : set){
            if(set.contains(x-1)) //如果存在比他小1的值，则考虑比他小的值即可。
                continue;
            int t = x; //不存在比他小1的值，说明当前值可能是最长连续序列的起点
            while(set.contains(t))
                t++;
            maxLength = Math.max(maxLength, t - x); //更新答案
        }
        return maxLength;
    }
}
```

# 双指针

## [1.移动零](https://leetcode.cn/problems/move-zeroes/description/)
思路：使用双指针，左指针始终指向第一个0， 右指针指向left后的第一个非零元素（其中left到right-1都为0），进行元素交换后，双指针继续向后移动。  
代码：
```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0; //始终指向第一个0
        while(left < n && nums[left] != 0)
            left++;
        int right = left; //指向left后的非零元素
        while(right < n && nums[right] == 0)
            right++;
        while(right < n){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            while(right < n && nums[right] == 0)
                right++;
            while(left < n && nums[left] != 0)
                left++;
        }
    }
}
```

## [2.盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)
思路：定义两个指针left和right，left从最左边开始，right从最右边开始，每次移动最矮的那条边（因为移动最矮边才可能在中间找到最优解），并记录储存的最大水量。  
代码：
```
class Solution {
    public int maxArea(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length - 1;
        while(left < right){
            int s = (right - left) * Math.min(height[left], height[right]);
            ans = Math.max(ans, s);
            if(height[left] < height[right])
                left++;
            else
                right--;
        }
        return ans;
    }
}
```

## [3.三数之和](https://leetcode.cn/problems/3sum/description/)
思路：先排序，然后定义三个指针i，left和right，外层循环指针i从0遍历到倒数第三个位置（留两个位置给相向指针left和right），指针left从i+1开始，指针right从最右边开始。（如果题目要求结果不能重复，则三个指针都只指向第一个重复元素，绕过后续重复元素）  
代码：
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); //排序
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        int i = 0;
        while(i < n - 2){
            int left = i + 1;
            int right = n - 1;
            while(left < right){
                if(nums[i] + nums[left] + nums[right] == 0){
                    ans.add(new ArrayList(Arrays.asList(nums[i], nums[left], nums[right])));
                    left++;
                    while(left < right && nums[left-1] == nums[left])//指向第一个重复的元素
                        left++;
                    right--;
                    while(left < right && nums[right] == nums[right+1])//指向第一个重复的元素
                        right--;
                }
                else if(nums[i] + nums[left] + nums[right] < 0)
                    left++;
                else
                    right--;
            }
            i++;
            while(i < n - 2 && nums[i-1] == nums[i]) //指向第一个重复的元素,绕过后续重复元素
                i++;
        }
        return ans;
    }
}
```

## [4.接雨水](https://leetcode.cn/problems/trapping-rain-water/description/)
思路：定义两个数组，前缀数组记录从左边到下标i的最大高度，后缀数组记录从右边到下标i的最大高度，有了这两个数组就可以计算各个下标i的盛水单位量。  
代码：
```
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] pre = new int[n];
        int[] post = new int[n];
        //求前缀数组和后缀数组
        int max = 0;
        for(int i = 0; i < n; i++){
            max = Math.max(max, height[i]);
            pre[i] = max;
        }
        max = 0;
        for(int i = n-1; i >= 0; i--){
            max = Math.max(max, height[i]);
            post[i] = max;
        }
        //根据前缀数组和后缀数组求答案
        int ans = 0;
        for(int i = 0; i < n; i++){
            ans += Math.min(pre[i], post[i]) - height[i];
        }
        return ans;
    }
}
```

# 滑动窗口

### 模板代码
```
int start = 0;
int end = 0;
while(end < n){ //外层循环扩展右边界，内层循环扩展左边界
    xxx;  //当前考虑的元素
    while(start <= end && 条件){
        xxx
        start++; //扩展左边界
    }
    end++; //扩展右边界
}
```

### 滑动窗口使用前提：  
1. 连续子数组。
2. 有单调性。比如元素均为正数。

## [1.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
思路：套用滑动窗口模板  
代码：
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        char[] strArr = s.toCharArray();
        int n = strArr.length;
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while(right < n){
            char c = strArr[right];
            while(left <= right && set.contains(c)){
                set.remove(strArr[left]);
                left++;
            }
            set.add(c);
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        return maxLength;
    }
}
```

## [2.找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/)
思路：首先统计字符串p的各个字母出现的次数，然后套用滑动窗口模板，统计滑动窗口中各字符的次数，如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边。当滑动窗口的长度等于字符串p的长度时，即找到一个答案。之后继续查找。  
代码：
```
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ans = new ArrayList<>();
        //统计字符串p的各个字母出现的次数
        char[] p_mp = new char[26];
        char[] pArr = p.toCharArray();
        for(char c : pArr){
            p_mp[c-'a']++;
        }
        //找到异位词
        char[] s_mp = new char[26];
        int left = 0;
        int right = 0;
        char[] sArr = s.toCharArray();
        int n = sArr.length;
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            // s_mp.merge(c, 1, Integer::sum); 
            s_mp[c-'a']++; //添加到hashmap
            //如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边
            while(left <= right && s_mp[c-'a'] > p_mp[c-'a']){ 
                s_mp[sArr[left]-'a']--;
                left++;
            }
            if(right - left + 1 == p.length()) //找到答案
                ans.add(left);
            right++;
        }
        return ans;
    }
}
```

# 子串

## [1.和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/)
思路：前缀和pre[i]表示[0...i]的和。和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]。因此转换思路，在for循环中，查找以i结尾的和为k的子数组个数时，只需计算前面有多少个和为pre[i] - k的子数组。  
代码：
```
class Solution {
    public int subarraySum(int[] nums, int k) {
        //前缀和pre[i]表示[0...i]的和
        //和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]
        //因此for循环，查找以i结尾的和为k的子数组个数，只需计算前面有多少个和为pre[i] - k的子数组
        int ans = 0; //记录答案
        int pre = 0; //记录前缀和
        Map<Integer, Integer> mp = new HashMap<>(); //记录前缀和的出现个数
        mp.put(0, 1); //初始化，pre[-1] = 0;
        for(int i = 0; i < nums.length; i++){
            pre += nums[i]; //更新前缀和
            if(mp.containsKey(pre - k))
                ans += mp.get(pre - k);
            mp.merge(pre, 1, Integer::sum);
        }
        return ans;
    }
}
```

## [2.滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)
思路：从左到右，维护一个单调队列，队首元素就是当前滑动窗口的最大值，详见代码注释。  
代码：
```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> dequeue = new LinkedList<>(); //单调队列
        for(int i = 0; i < k-1; i++){ //初始维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
        }
        //移动滑动窗口得到答案
        int n = nums.length;
        int[] ans = new int[n-k+1];
        for(int i = k-1; i < n; i++){
            //维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
            //检查队首是否已经超出滑动窗口了
            while(dequeue.getFirst() < i - k + 1)
                dequeue.removeFirst();
            //队首元素即为当前滑动窗口最大值
            ans[i-k+1] = nums[dequeue.getFirst()];
        }
        return ans;
    }
}
```


## [3.最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/)
思路：属于滑动窗口专题，和滑动窗口专题的第二题很相似。虽然存在双重循环，但时间复杂度为并不是O(n^2)，而是O(C·n)，其中C是一个常数。因为这里内层循环遍历的是52个大小写字母，是固定的常数。
代码：
```
class Solution {
    public String minWindow(String s, String t) {
        char[] t_mp = new char[128]; //t的哈希表
        char[] tArr = t.toCharArray();
        for(char c : tArr){
            t_mp[c - 'A']++;
        }
        char[] s_mp = new char[128]; //s的哈希表
        char[] sArr = s.toCharArray();
        int left = 0;
        int right = 0;
        int n = sArr.length;
        int[] ans = new int[]{0, 100001}; //记录最小覆盖子串的起始下标和长度
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            s_mp[c - 'A']++; //加入hash表
            while(left <= right && isCover(t_mp, s_mp)){
                if(right - left + 1 < ans[1]){ //更新并记录答案
                    ans[0] = left;
                    ans[1] = right - left + 1;
                }
                s_mp[sArr[left] - 'A']--;
                left++;
            }
            right++;
        }
        if(ans[1] > 100000) return "";
        else return s.substring(ans[0], ans[0]+ans[1]);
    }

    //遍历52个字母，检查当前滑动窗口是否覆盖了字符串t
    boolean isCover(char[] t_mp, char[] s_mp){
        for(int i = 0; i < 26; i++) //大写字母
            if(t_mp[i] > s_mp[i])
                return false;
        for(int i = 0; i < 26; i++) //小写字母
            if(t_mp[i+'a'-'A'] > s_mp[i+'a'-'A'])
                return false;
        return true;
    }
}
```

# 普通数组

## [1.最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

思路1：套用滑动窗口模板。  
代码：
```
class Solution {
    public int maxSubArray(int[] nums) {
        int ans = Integer.MIN_VALUE;
        int left = 0;
        int right = 0;
        int sum = 0;
        while(right < nums.length){
            sum += nums[right];
            ans = Math.max(sum, ans); //更新答案
            while(left <= right && sum <= 0){
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return ans;
    }
}
```

思路2：前缀和做法。由于子数组的元素和等于两个前缀和的差。我们可以一边遍历数组计算前缀和，一边维护前缀和的最小值，当前的前缀和减去前缀和的最小值，就得到了以当前元素结尾的子数组和的最大值，用它来更新答案的最大值。   
代码：  
```
class Solution {
    public int maxSubArray(int[] nums) {
        int pre = 0; //记录以下标i结果的前缀和
        int min_pre = 0; //记录i之前的最小前缀和
        int ans = Integer.MIN_VALUE; //记录答案
        for(int i = 0; i < nums.length; i++){
            pre += nums[i]; //计算前缀和
            ans = Math.max(pre - min_pre, ans); //更新答案
            min_pre = Math.min(min_pre, pre); //更新最小前缀和
        }
        return ans;
    }
}
```

## [2.合并区间](https://leetcode.cn/problems/merge-intervals/description/)

思路：对所有区间按照左端点从小到大进行排序。for循环，如果不相交无法合并则加入答案，如果相交可以合并则更新答案。注意List到Array数组的转换代码。  
代码：
```
class Solution {
    public int[][] merge(int[][] intervals) {
        List<int[]> ans = new ArrayList<>();
        //按照区间左端点进行排序
        Arrays.sort(intervals, (p, q)->{
            return p[0] - q[0];
        });
        for(int i = 0; i < intervals.length; i++){
            if(i != 0 && intervals[i][0] <= ans.get(ans.size()-1)[1]){// 可以合并
                //更新右端点
                ans.get(ans.size()-1)[1] = Math.max(intervals[i][1], ans.get(ans.size()-1)[1]);
            }
            else ans.add(intervals[i]); // 不相交，无法合并
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
```

## [3.轮转数组](https://leetcode.cn/problems/rotate-array/description/)

思路：反转整个数组，然后反转前k个元素，最后反转剩余元素，即可得出答案。注意k取模，防止k大于n。
代码：
```
class Solution {
    public void rotate(int[] nums, int k) {
        //一种做法是需要一个额外数组，考虑原地轮转的方法
        int n = nums.length;
        k = k % n; //k取模，防止k大于n
        reverse(nums, 0, n-1); //反转整个数组
        reverse(nums, 0, k-1); //反转前k个元素
        reverse(nums, k, n-1); //反转剩余元素
    }

    public void reverse(int[] nums, int l, int r){ //反转特定区间内的元素
        while(l <= r){
            int t = nums[l];
            nums[l] = nums[r];
            nums[r] = t;
            l++;
            r--;
        }
    }
}
```

## [4.除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/)

思路：利用前缀积和后缀积即可得到答案。  
代码：
```
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] pre = new int[n]; //pre[i]表示i之前所有元素的前缀积
        int[] post = new int[n]; //post[i]表示i之后所有元素的后缀积
        //计算前缀积和后缀积
        pre[0] = 1;
        for(int i = 1; i < n; i++)
            pre[i] = pre[i-1] * nums[i-1];
        post[n-1] = 1;
        for(int i = n-2; i >= 0; i--)
            post[i] = post[i+1] * nums[i+1];
        //计算答案
        int[] ans = new int[n];
        for(int i = 0; i < n; i++)
            ans[i] = pre[i] * post[i];
        return ans;
    }
}
```

# 矩阵

## [1.矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/description/)

思路：我们可以用两个标记数组分别记录每一行和每一列是否有零出现。    
代码：
```
class Solution {
    public void setZeroes(int[][] matrix) {
        int[] row = new int[matrix.length]; //记录每行是否有0
        int[] col = new int[matrix[0].length]; //记录每列是否有0
        //记录各行各列的0情况
        for(int i = 0; i < row.length; i++){
            for(int j = 0; j < col.length; j++){
                if(matrix[i][j] == 0){
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }
        //原地修改
        for(int i = 0; i < row.length; i++){
            for(int j = 0; j < col.length; j++){
                if(row[i] == 1 || col[j] == 1)
                    matrix[i][j] = 0;
            }
        }
    }
}
```

## [2.螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/)

思路：定义“右下左上”四个方向，初始为右方向，一直走，直到越界或者格子已访问，则转换下一个方向，重复此过程。 当走过所有格子则结束。   
代码：
```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int[][] dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};//右下左上
        int curDir = 0; //当前方向
        int row = matrix.length;
        int col = matrix[0].length;
        List<Integer> ans = new ArrayList<>();//答案
        int curX = 0; //当前坐标
        int curY = 0; //当前坐标
        int nums = 0; //表示走过的格子数
        ans.add(matrix[curX][curY]);//记录答案
        matrix[curX][curY] = 101; //标记当前位置已经走过
        nums++;
        while(nums < row * col){ //判断是否走完所有格子
            int nextX = curX + dir[curDir][0];
            int nextY = curY + dir[curDir][1];
            //判断下一个位置是否越界或者已经访问过了
            if(nextX < 0 || nextX >= row || nextY < 0 || nextY >= col || matrix[nextX][nextY] > 100){
                curDir = (curDir + 1) % 4; //修改方向
                nextX = curX + dir[curDir][0]; //重新走下一步
                nextY = curY + dir[curDir][1];
            }
            curX = nextX;
            curY = nextY;
            ans.add(matrix[curX][curY]);//记录答案
            matrix[curX][curY] = 101; //标记当前位置已经走过
            nums++;
        }
        return ans;
    }
}
```

## [3.旋转图像](https://leetcode.cn/problems/rotate-image/description/)

思路：先以左对角线进行翻转，再水平翻转。   
代码：
```
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //1.先以左对角线进行翻转
        for(int i = 0; i < n; i++){
            for(int j = i; j < n; j++){
                int t = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = t;
            }
        }
        //2.再水平翻转
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n/2; j++){
                int t = matrix[i][j];
                matrix[i][j] = matrix[i][n-1-j];
                matrix[i][n-1-j] = t;
            }
        }
    }
}
```

## [4.搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/)

思路：右上角元素是二叉搜索树（灵神）。还有一种思路对每行进行二分查找。   
代码：
```
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        //右上角元素是二叉搜索树
        int x = 0;
        int y = n - 1;
        while(x < m && y >= 0){
            if(matrix[x][y] == target)
                return true;
            else if(matrix[x][y] > target) //左子树
                y--;
            else //右子树
                x++;
        }
        return false;
    }
}
```

# 链表

## [1.相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/)

思路：设a链表长为x+z，b链表长为y+z，其中z是公共链部分。a走完x+z再走y，b走完y+z再走x，如果两个链表相交，则会刚好相遇。      
代码：
```
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = headB;
        while(p1 != p2){
            if(p1 != null) p1 = p1.next;
            else p1 = headB;
            if(p2 != null) p2 = p2.next;
            else p2 = headA;
        }
        return p1;
    }
}
```

## [2.反转链表](https://leetcode.cn/problems/reverse-linked-list/description/)
思路：反转链表经典模板。使用两个指针pre和cur，反转结束后，从原来的链表上看，pre指向最末尾节点，cur指向后续的下一个节点，为null。  
代码：
```
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;
        while(cur != null)
        {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
```

## [3.回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/)
思路：快慢指针模板找到链表的中间节点 + 反转链表。  
代码：
```
class Solution {
    public boolean isPalindrome(ListNode head) {
        //利用快慢指针找到链表的中间节点
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        //找到中间节点，反转总链表的后半部分
        ListNode tail = reverse(slow);
        //检查是否为回文链表
        while(tail != null){
            if(head.val != tail.val)
                return false;
            head = head.next;
            tail = tail.next;
        }
        return true;
    }

    public ListNode reverse(ListNode node){
        ListNode pre = null;
        ListNode cur = node;
        while(cur != null){
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
```

## [4.环形链表](https://leetcode.cn/problems/linked-list-cycle/description/)
思路：快慢指针判断是否存在环。若fast指针达到结尾（使用模板判断，因为fast一次走两步），则不存在环。若存在环，则fast肯定会追上slow指针。    
代码：
```
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow) //若存在环，快指针一定会追上慢指针
                return true;
        }
        return false; //没有环
    }
}
```

## [5.环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)
思路：  
![image](https://github.com/user-attachments/assets/4af5834a-f78d-4eec-a66f-979847bb3022)  
![image](https://github.com/user-attachments/assets/732720b3-6468-4103-9552-1414b62fd844)  

代码：
```
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow){ //快指针追上慢指针，存在环
                while(head != slow){
                    head = head.next;
                    slow = slow.next;
                }
                return head; //找到开始入环的第一个节点
            }
        }
        return null; //不存在环
    }
}
```

## [6.合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description)
思路：空间复杂度为O(1)。设置一个哨兵节点，cur指针指向哨兵。同时有list1和list2双指针，cur每次指向双指针较小的那一个，以此类推。    
代码：
```
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode preHead = new ListNode(); //哨兵节点
        ListNode cur = preHead;
        while(list1 != null && list2 != null){
            if(list1.val < list2.val){
                cur.next = list1;
                cur = cur.next;
                list1 = list1.next;
            }
            else{
                cur.next = list2;
                cur = cur.next;
                list2 = list2.next;
            }
        }
        if(list1 == null){
            cur.next = list2;
        }else{
            cur.next = list1;
        }
        return preHead.next;
    }
}
```

## [7.两数相加](https://leetcode.cn/problems/add-two-numbers/description/)
思路：创建一个哨兵节点。然后同时遍历l1和l2，直到最长链为null且进位为0为止。每次遍历计算当前位和当前进位即可。    
代码：
```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //哨兵节点
        ListNode preHead = new ListNode();
        ListNode cur = preHead;
        int carry = 0; //保存当前进位
        while(l1 != null || l2 != null || carry != 0){
            int val1 = l1 != null ? l1.val : 0;
            int val2 = l2 != null ? l2.val : 0;
            int curVal = val1 + val2 + carry; //计算当前位
            cur.next = new ListNode(curVal % 10);//更新答案
            cur = cur.next;
            carry = curVal / 10; //进位
            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
        }
        return preHead.next;
    }
}
```

## [8.删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/)
思路：由于可能会删除链表头部，用哨兵节点简化代码。使用前后指针left和right。right指针先走n步，然后left指针和right指针同时走，距离始终为n，直到right指针指向最后一个节点，则left指针指向倒数第n+1个节点，删除倒数第n个节点。      
代码：
```
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode preHead = new ListNode(0, head); //哨兵节点
        ListNode slow = preHead;
        ListNode fast = preHead;
        for(int i = 0; i < n; i++) //fast指针先走n走
            fast = fast.next;
        while(fast.next != null){ //fast和slow指针再同时走
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next; //此时slow.next即为待删除节点，删除
        return preHead.next;
    }
}
```


## [9.两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/)
思路：从左到右遍历，每段（长度为2）进行链表反转。      
代码：
```
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode preHead = new ListNode(0, head); //哨兵节点
        ListNode preLeft = preHead; //用于连接交换后长度为2的链表
        ListNode cur = head;
        while(cur != null && cur.next != null){
            //两两交换
            ListNode pre = null;
            ListNode post = cur; //记录交换后长度为2的链表的尾节点
            for(int i = 0; i < 2; i++){
                ListNode nxt = cur.next;
                cur.next = pre;
                pre = cur;
                cur = nxt;
            }
            //重新拼接
            preLeft.next = pre;
            post.next = cur;
            preLeft = post;
        }
        return preHead.next;
    }
}
```

## [10.随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/description)
思路1：可以遍历两次链表：第一次是为了创建所有节点并存储在哈希表中；第二次则是为了设置这些新节点的 next 和 random 指针。（空间复杂度为O(n)）      
代码：
```
class Solution {
    public Node copyRandomList(Node head) {
        Map<Node, Node> mp = new HashMap<>();//存放新老链表的映射
        Node cur = head;
        //准备工作
        while(cur != null){
            mp.put(cur, new Node(cur.val)); //复制新节点，并和老节点对应
            cur = cur.next;
        }
        //深拷贝
        cur = head;
        while(cur != null){
            Node new_cur = mp.get(cur);
            if(cur.next != null) new_cur.next = mp.get(cur.next);
            if(cur.random != null) new_cur.random = mp.get(cur.random);
            cur = cur.next;
        }
        return mp.get(head);
    }
}
```

思路2：生成交错链表，老->新->老->新...，然后遍历交错链表，复制新节点的random，最后遍历交错链表，拆分出新链表和老链表。（空间复杂度为O(1)）   
代码：
```
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        //生成交错链表，老->新->老->新...
        while(cur != null){
            Node new_node = new Node(cur.val);//创建新节点
            new_node.next = cur.next; //插入新节点
            cur.next = new_node;
            cur = cur.next.next;
        }
        //遍历交错链表，复制新节点的random
        cur = head;
        while(cur != null){
            Node new_node = cur.next;
            if(cur.random != null) new_node.random = cur.random.next;
            cur = cur.next.next;
        }
        //遍历交错链表，拆分出新链表和老链表（答案会检测老链表是否被修改）
        cur = head;
        Node preAns = new Node(0); //答案的哨兵节点
        Node curCopy = preAns;
        while(cur != null){
            curCopy.next = cur.next;
            curCopy = curCopy.next;
            cur.next = cur.next.next;
            cur = cur.next;
        }
        return preAns.next;
    }
}
```

## [11.排序链表](https://leetcode.cn/problems/sort-list/description)
思路：归并排序思想，详见代码注释。       
代码：
```
class Solution {
    //归并排序(分而治之)
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null) //0或1个节点则有序，返回
            return head;
        ListNode midNode = findMidNode(head);
        ListNode l1 = sortList(head); //排序前半部分
        ListNode l2 = sortList(midNode); //排序后半部分
        return merge(l1, l2); //合并两个有序链表
    }

    //快慢指针找到中间节点，并将链表分为前后两半
    public ListNode findMidNode(ListNode head){
        ListNode fast = head;
        ListNode slow = head;
        ListNode slowPre = null;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slowPre = slow;
            slow = slow.next;
        }
        slowPre.next = null; //中间节点的前缀节点的next指向null
        return slow;
    }

    //合并两个有序链表
    public ListNode merge(ListNode l1, ListNode l2){
        ListNode preHead = new ListNode(0); //哨兵节点
        ListNode cur = preHead;
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else{
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if(l1 == null)
            cur.next = l2;
        else
            cur.next = l1;
        return preHead.next;
    }
}
```


## [12.LRU 缓存](https://leetcode.cn/problems/lru-cache/description/)
思路：使用双向链表（有一个哨兵节点来简化操作）来模拟LRU缓存。每次新增节点时，则将其插入到哨兵节点之后（表示最近访问，若超出容量则删除双向链表最后一个节点）；更新或者查询节点时，则将原先节点移到哨兵节点之后（表示最近访问）。详见代码注释。       
代码：
```
class LRUCache {

    class Node { //存放的节点
        int key, value;
        Node pre, next;

        public Node(){
            this.key = -1;
            this.value = -1;
            this.pre = null;
            this.next = null;
        }
        public Node(int key, int value, Node pre, Node next){
            this.key = key;
            this.value = value;
            this.pre = pre;
            this.next = next;
        }
    }

    int capacity; //容量
    int nums; //当前LRU缓存数
    Map<Integer, Node> mp = new HashMap<>(); //用于快速找到key对应的节点
    Node preHead;//哨兵节点，LRU缓存用双向链表模拟

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.nums = 0;
        preHead = new Node(); //初始化双向链表
        preHead.pre = preHead;
        preHead.next = preHead;
    }
    
    public int get(int key) {
        //首先判断LRU是否有该key
        if(mp.containsKey(key)){
            Node node = mp.get(key); //获得查询节点
            //删除该节点
            node.pre.next = node.next;
            node.next.pre = node.pre;
            //将该节点移动到哨兵结点之后，表示最新访问
            node.next = preHead.next;
            node.pre = preHead;
            preHead.next.pre = node;
            preHead.next = node;
            return node.value;
        }
        else    
            return -1;
    }
    
    public void put(int key, int value) {
        //首先判断LRU是否有该key
        if(mp.containsKey(key)){ //如果关键字 key 已经存在
            Node node = mp.get(key);
            node.value = value; //则变更其数据值 value
            //删除该节点
            node.pre.next = node.next;
            node.next.pre = node.pre;
            //将该节点移动到哨兵结点之后，表示最新访问
            node.next = preHead.next;
            node.pre = preHead;
            preHead.next.pre = node;
            preHead.next = node;
        }else{
            //新节点插入到哨兵结点之后
            Node newNode = new Node(key, value, preHead, preHead.next); 
            preHead.next.pre = newNode;
            preHead.next = newNode;
            mp.put(key, newNode);
            nums++;
            //检查容量
            if(nums > capacity){
                //删除双向链表最后一个节点
                Node delNode = preHead.pre;
                delNode.pre.next = preHead;
                preHead.pre = delNode.pre;
                nums--;
                mp.remove(delNode.key);
            }
        }
    }
}
```

# 二叉树

## [1.二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/)
思路1：递归。       
代码：
```
class Solution {
    List<Integer> ans = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        dfs(root);
        return ans;
    }

    public void dfs(TreeNode node){
        if(node == null)
            return;
        dfs(node.left);
        ans.add(node.val);
        dfs(node.right);
    }
}
```

思路2：迭代，两种方式是等价的，区别在于递归的时候隐式地维护了一个栈，而我们在迭代的时候需要显式地将这个栈模拟出来，其他都相同，具体实现可以看下面的代码。。       
代码：
```
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        List<TreeNode> stack = new LinkedList<>(); //栈
        while(root != null || !stack.isEmpty()){ //root表示当前节点
            while(root != null){ //一直深入左子树
                stack.addLast(root); //入栈
                root = root.left;
            }
            root = stack.getLast(); //取出栈顶元素
            stack.removeLast();
            ans.add(root.val); //保存答案
            root = root.right; //继续查找右子树
        }
        return ans;
    }
}
```

## [2.二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description)
思路：当前子树的最大深度 = max（左子树的最大深度，右子树的最大深度）+ 1。       
代码：
```
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

## [3.翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description)
思路：先递归翻转左右子树，再交换左右子树。也可以先交换左右子树，再递归翻转左右子树。       
代码：
```
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null)
            return null;
        TreeNode lTree = invertTree(root.left);//翻转后的左子树
        TreeNode rTree = invertTree(root.right); //翻转后的右子树
        root.left = rTree; //交换左右子树
        root.right = lTree;
        return root;
    }
}
```

## [4.对称二叉树](https://leetcode.cn/problems/symmetric-tree/)
思路：两个子树是否轴对称 = 根节点是否相等 + 左右子树是否轴对称 + 右左子树是否轴对称。      
代码：
```
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return dfs(root.left, root.right);
    }

    public boolean dfs(TreeNode tree1, TreeNode tree2){ //判断两棵子树是否镜像对称
        if(tree1 == null && tree2 == null) //都为null
            return true;
        if(tree1 == null || tree2 == null) //有一个为null
            return false;
        return tree1.val == tree2.val && dfs(tree1.left, tree2.right) && dfs(tree1.right, tree2.left);
    }
}
```

## [5.二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/)
思路：树形DP。类似于求树的最大深度，求树的最大链长（最大深度-1）。遍历每个节点时，顺便计算经过当前节点的最大直径 = 左子树链长 + 右子树链长 + 2.   
代码：
```
class Solution {
    public int ans = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        maxLength(root);
        return ans;
    }

    public int maxLength(TreeNode node){ //返回树的最大链长（最大深度-1）
        if(node == null)
            return -1;
        int lLength = maxLength(node.left);
        int rLength = maxLength(node.right);
        ans = Math.max(ans, lLength + rLength + 2); //更新答案
        return Math.max(lLength, rLength) + 1;
    }
}
```


## [6.二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)

### BFS模板
```
首元素入队
while(队列不为空)
{
  s=弹出队头元素。
  print（s）
  s的所有节点入队
}
```

思路：二叉树的BFS。当队列不为空时，队列的元素个数就是当前层的节点数，详见代码。     
代码：
```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<TreeNode> queue = new LinkedList<>();//队列
        List<List<Integer>> ans = new ArrayList<>();
        if(root == null) return ans;
        //根节点先入队
        queue.addLast(root);
        //BFS
        while(!queue.isEmpty()){
            List<Integer> list = new ArrayList<>();
            int n = queue.size(); //队列元素个数表示当前层有几个节点
            for(int i = 0; i < n; i++){
                TreeNode node = queue.getFirst();
                queue.removeFirst();
                list.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            ans.add(list);
        }
        return ans;
    }
}
```

## [7.将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description)
思路：二叉搜索树的中序遍历是升序序列，题目给定的数组是按照升序排序的有序数组，因此可以确保数组是二叉搜索树的中序遍历序列。其中一种方法是中序遍历，总是选择中间位置左边的数字作为根节点。   
代码：
```
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return func(nums, 0, nums.length-1);
    }

    public TreeNode func(int[] nums, int left, int right){
        if(left > right)
            return null;
        int mid = (left + right) / 2;
        TreeNode lTree = func(nums, left, mid-1);
        TreeNode rTree = func(nums, mid+1, right);
        return new TreeNode(nums[mid], lTree, rTree);
    }
}
```

## [8.验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

### 方法一（前序遍历）
思路：每次维护一个开区间，判断当前节点值是否在该区间内。   
代码：
```
class Solution {
    public boolean isValidBST(TreeNode root) {
        return preDfs(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean preDfs(TreeNode node, long left, long right) //开区间(left, right)
    {
        if(node == null)
            return true;
        return left < node.val && node.val < right && 
                preDfs(node.left, left, node.val) && 
                preDfs(node.right, node.val, right);
    }
}
```

### 方法二（中序遍历）
思路：如果树是一个二叉搜索树,则中序遍历得到的序列是递增的。可以维护一个值pre,用于记录上一个节点的值.    
代码：
```
class Solution {
    public long pre = Long.MIN_VALUE;

    public boolean isValidBST(TreeNode root) {
        if(root == null)
            return true;
        if(!isValidBST(root.left) || root.val <= pre)
            return false;
        pre = root.val;
        return isValidBST(root.right);
    }
}
```

### 方法三（后序遍历）
思路：每次返回左右子树的最大最小值区间，判断当前节点值是否符合条件.     
代码：
```
class Solution {
    public boolean isValidBST(TreeNode root) {
        return dfs(root)[0] != Long.MIN_VALUE;
    }

    public long[] dfs(TreeNode node){ //每次返回当前子树的最小和最大值
        if(node == null)
            return new long[]{Long.MAX_VALUE, Long.MIN_VALUE};
        long[] lTree = dfs(node.left);
        long[] rTree = dfs(node.right);
        if(lTree[1] < node.val && node.val < rTree[0]) //满足二叉搜索树
            //max和min适用于当前节点的子树至少有一个为null的情况
            return new long[]{Math.min(lTree[0], node.val), Math.max(rTree[1], node.val)};
        else //不满足二叉搜索树
            return new long[]{Long.MIN_VALUE, Long.MAX_VALUE};
    }
}
```

## [9.二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/)
思路：中序遍历，见代码注释。也可以直接用kthSmallest函数做递归，返回-1表示还没找到。      
代码：
```
class Solution {
    public int n = 0; //当前找到的第n小元素
    public int kthSmallest(TreeNode root, int k) {
        return midDfs(root, k).val;
    }

    public TreeNode midDfs(TreeNode node, int k){
        if(node == null)
            return null;
        TreeNode lTree = midDfs(node.left, k);//查找左子树
        if(lTree != null) //左子树找到，直接返回
            return lTree;
        n++;
        if(n == k) //若当前节点是第k小元素
            return node;
        else
            return midDfs(node.right, k);//只会在右子树
    }
}
```

## [10.二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/description/)
思路：答案为每一层的最右节点，所以dfs优先递归右子树，如果当前节点深度大于答案的长度，说明该节点是该层的最右节点，加入答案；否则则相当于被右边的节点挡住了。其实用bfs层序遍历更直观。        
代码：
```
class Solution {
    public List<Integer> ans = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 1);
        return ans;
    }

    public void dfs(TreeNode node, int depth){
        if(node == null)
            return;
        if(depth > ans.size())
            ans.add(node.val);
        dfs(node.right, depth + 1);
        dfs(node.left, depth + 1);
    }
}
```


## [11.二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description)
思路：为了原地算法修改这棵树，可以采用逆向思维，从后往前构造这个树，这样可以防止从前往后构造时会丢失子树的地址。即按照右子树-左子树-根节点顺序遍历去构造（和前序遍历相反：根节点-左子树-右子树）        
代码：
```
class Solution {
    public TreeNode preHead = new TreeNode(0, null, null); //哨兵节点，用于头插法
    //右子树-左子树-根节点（和前序遍历相反：根节点-左子树-右子树）
    public void flatten(TreeNode root) {
        if(root == null)
            return;
        flatten(root.right);
        flatten(root.left);
        //头插法
        root.left = null;
        root.right = preHead.right;
        preHead.right = root;
    }
}
```


## [12.从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)
思路：前序序列：[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]， 中序序列：[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]。      
代码：
```
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return preAndInBuildTree(preorder, 0, preorder.length-1, inorder, 0, inorder.length-1);
    }

    public TreeNode preAndInBuildTree(int[] preorder, int prel, int prer, int[] inorder, int inl, int inr){
        if(prel > prer || inl > inr)
            return null;
        int inRootIdx = findMid(inorder, inl, inr, preorder[prel]); //找到根节点在inorder中的下标
        int lNum = inRootIdx - inl;//左子树节点数
        int rNum = inr - inRootIdx; //右子树节点数
        TreeNode lTree = preAndInBuildTree(preorder, prel+1, prel+lNum, inorder, inl, inRootIdx-1);
        TreeNode rTree = preAndInBuildTree(preorder, prer-rNum+1, prer, inorder, inRootIdx+1, inr);
        return new TreeNode(preorder[prel], lTree, rTree);
    }

    public int findMid(int[] nums, int inl, int inr, int x){ //返回x在nums数组的下标，可以用哈希表优化时间
        for(int i = inl; i <= inr; i++)
            if(nums[i] == x)
                return i;
        return -1;
    }
}
```

## [13.二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)
思路：![image](https://github.com/user-attachments/assets/673ddc41-6755-4dc7-b9ef-b505546b984a)
![image](https://github.com/user-attachments/assets/3f405ffc-df86-46cc-98b5-1c33a2f7199e)

代码：
```
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null)
            return null;
        if(root == p || root == q)
            return root;
        //首先dfs递归找到p，q节点并返回
        TreeNode lTree = lowestCommonAncestor(root.left, p, q);
        TreeNode rTree = lowestCommonAncestor(root.right, p, q);
        //分类讨论返回值
        if(lTree != null && rTree != null)// 左右都找到
            return root; // 当前节点是最近公共祖先
        else if(lTree == null && rTree == null)
            return null;
        else if(lTree != null)
            return lTree;
        else
            return rTree;
    }
}
```


## [14.路径总和 III](https://leetcode.cn/problems/path-sum-iii/description)
思路：每次自顶向下寻找符合条件的路径和时，思路和前面[和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/)一样，区别在于需要回溯哈希表（因为当我们递归完左子树，要递归右子树时，map中还保存着左子树的数据。但递归到右子树，要计算的路径并不涉及到左子树的任何节点）。        
代码：
```
class Solution {
    public int ans = 0;
    public Map<Long, Integer> mp = new HashMap<>();//记录前缀和的频数
    public int pathSum(TreeNode root, int targetSum) {
        mp.put(0L, 1); //初始化
        dfs(root, 0, targetSum);
        return ans;
    }
    
    public void dfs(TreeNode node, long sum, int targetSum){
        if(node == null)
            return;
        sum += node.val;
        if(mp.containsKey(sum - targetSum)) //检查前面的前缀和
            ans += mp.get(sum - targetSum);
        mp.merge(sum, 1, Integer::sum);//记录前缀和
        dfs(node.left, sum, targetSum);
        dfs(node.right, sum, targetSum);
        //回溯 mp.merge(sum, -1, Integer::sum);
        if(mp.get(sum) > 1) mp.put(sum, mp.get(sum)-1);
        else mp.remove(sum);
    }
}
```

# 回溯

## [1.子集](https://leetcode.cn/problems/subsets/description/)
思路：子集型。每个元素可选可不选，需要回溯。     
代码：
```
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        dfs(nums, new ArrayList<>(), 0);
        return ans;
    }

    public void dfs(int[] nums, List<Integer> sub, int depth){
        if(depth == nums.length){
            ans.add(new ArrayList<>(sub)); //记得拷贝
            return;
        }
        //选
        sub.add(nums[depth]);
        dfs(nums, sub, depth + 1);
        sub.remove(sub.size()-1); //回溯
        //不选
        dfs(nums, sub, depth + 1);
    }
}
```

## [2.全排列](https://leetcode.cn/problems/permutations/description/)
思路：排列型。用一个boolean数组记录该元素是否已经被选过了。       
代码：
```
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        dfs(nums, new boolean[nums.length], new ArrayList<>());
        return ans;
    }

    public void dfs(int[] nums, boolean[] isSelect, List<Integer> arr){
        if(arr.size() == nums.length){
            ans.add(new ArrayList<>(arr));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(!isSelect[i]){
                isSelect[i] = true;
                arr.add(nums[i]);
                dfs(nums, isSelect, arr);
                isSelect[i] = false; //回溯
                arr.remove(arr.size()-1);//回溯
            }
        }
    }
}
```

## [3.电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/)
思路：经典dfs题目。用char[]替代String可以显著提升时间。     
代码
```
class Solution {
    Map<Character, String> mp = new HashMap<>();
    List<String> ans = new ArrayList<>();

    public List<String> letterCombinations(String digits) {
        if(digits.equals("")) return ans;
        mp.put('2', "abc");
        mp.put('3', "def");
        mp.put('4', "ghi");
        mp.put('5', "jkl");
        mp.put('6', "mno");
        mp.put('7', "pqrs");
        mp.put('8', "tuv");
        mp.put('9', "wxyz");
        dfs(digits, new char[digits.length()], 0);
        return ans;
    }

    public void dfs(String digits, char[] str, int i){
        if(i == digits.length()){
            ans.add(new String(str));
            return;
        }
        String cur_str = mp.get(digits.charAt(i));
        for(int j = 0; j < cur_str.length(); j++){
            str[i] = cur_str.charAt(j);
            dfs(digits, str, i+1);
        }
    }
}
```

## [4.组合总和](https://leetcode.cn/problems/combination-sum/description/)
思路：每个元素可选（一个或者多个）可不选，需要回溯。见代码注释。       
代码：
```
class Solution {
    public List<List<Integer>> ans = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        dfs(candidates, target, new ArrayList<>(), 0, 0);
        return ans;
    }

    public void dfs(int[] candidates, int target, List<Integer> arr, int i, int sum){
        if(i == candidates.length){
            if(sum == target)
                ans.add(new ArrayList<>(arr));
            return;
        }
        //选（一个或者多个）
        int n = (target - sum) / candidates[i];  //第i个元素最多可以选n个
        for(int j = 0; j < n; j++){
            arr.add(candidates[i]);
            sum += candidates[i];
            dfs(candidates, target, arr, i+1, sum);
        }
        for(int j = 0; j < n; j++){ //回溯
            arr.remove(arr.size()-1);
            sum -= candidates[i];
        }
        //不选
        dfs(candidates, target, arr, i+1, sum);
    }
}
```

## [5.括号生成](https://leetcode.cn/problems/generate-parentheses/description/)
思路：答案固定为一个长度2*n的字符串，对于每个位置i，选左括号还是不选左括号（选右括号）。当左括号数目小于n，则可以选左括号；当右括号数目小于左括号，则可以选右括号；       
代码：
```
class Solution {
    List<String> ans = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        dfs(new char[n * 2], 0, 0, n);
        return ans;
    }

    public void dfs(char[] str, int left, int i, int n){
        if(i == 2 * n){
            ans.add(new String(str));
            return;
        }
        if(left < n){ //左括号个数还没满就可以选
            str[i] = '(';
            dfs(str, left + 1, i + 1, n);
        }
        if(i - left < left){ //右括号个数只要少于左括号就可以选
            str[i] = ')';
            dfs(str, left, i + 1, n);
        }
    }
}
```

## [6.分割回文串](https://leetcode.cn/problems/palindrome-partitioning/description/)
思路：假设每对相邻字符之间有个逗号，那么就看每个逗号是选还是不选。     
代码：
```
class Solution {
    List<List<String>> ans = new ArrayList<>();

    public List<List<String>> partition(String s) {
        dfs(new ArrayList<>(), 0, 0, s);
        return ans;
    }

    void dfs(List<String> path, int start, int i, String s)
    {
        if(i == s.length() - 1) //最后一个索引一定得切割
        {
            String fstr = s.substring(start, i+1);
            if(huiwen(fstr)) 
            {
                path.add(fstr);
                ans.add(new ArrayList<>(path)); //记得拷贝
                path.remove(path.size()-1);//回溯
            }
            return;
        }
        //索引i切割（子串包含索引i字符）(也可以看作相邻字符间的逗号)
        String str = s.substring(start, i+1);
        if(huiwen(str)) //若不符合，则剪枝
        {
            path.add(str);
            dfs(path, i+1, i+1, s);
            path.remove(path.size()-1);//回溯
        }
        //索引i不切割
        dfs(path, start, i+1, s);
    }

    //判断子串是否为都为回文串
    public boolean huiwen(String s)
    {
        char[] subs = s.toCharArray();
        int l = 0;
        int r = subs.length - 1;
        while(l <= r)
        {
            if(subs[l] != subs[r])
                return false;
            l++;
            r--;
        }
        return true;
    }
}
```

## [7.单词搜索](https://leetcode.cn/problems/word-search/description)
思路：经典dfs+回溯。再加两个优化，详见代码注释。     
代码：
```
class Solution {
    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length;
        //优化一:如果 word 的某个字母的出现次数，比 board 中的这个字母的出现次数还要多，可以直接返回 false。
        char[] mp = new char[128];
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                mp[board[i][j]]++;
        char[] wordc = word.toCharArray();
        char[] word_mp = new char[128];
        for(char c : wordc){
            word_mp[c]++;
            if(word_mp[c] > mp[c]) return false;
        }

        //优化二：检查word首和尾的字符，从在borad出现次数较少的那一个开始查找会更快
        if(mp[wordc[0]] > mp[wordc[wordc.length-1]])
            wordc = new StringBuilder(word).reverse().toString().toCharArray();
        
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                if(board[i][j] == wordc[0]){
                    char temp = board[i][j];
                    board[i][j] = '#';
                    if(dfs(board, wordc, i, j, 1)) return true;
                    board[i][j] = temp;
                }
        return false;
    }

    public boolean dfs(char[][] board, char[] word, int x, int y, int i){
        if(i == word.length){
            return true;
        }
        int[][] dirt = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        int m = board.length, n = board[0].length;
        for(int j = 0; j < 4; j++){
            int next_x = x + dirt[j][0], next_y = y + dirt[j][1];
            //next元素不越界，没被访问，满足下一个字符。不满足其中一个则剪枝。
            if(0 <= next_x && next_x < m && 0 <= next_y && next_y < n && 
            board[next_x][next_y] != '#' && board[next_x][next_y] == word[i]){
                board[next_x][next_y] = '#'; //标记已访问
                if(dfs(board, word, next_x, next_y, i + 1)) return true;//找到了
                board[next_x][next_y] = word[i]; //回溯
            }
        }
        return false;
    }
}
```

# 二分查找

## [1.搜索插入位置](https://leetcode.cn/problems/search-insert-position/description)
思路：套用二分查找模板即可。     
代码：
```
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }
}
```

### 二分查找模板代码
```
// lowerBound 返回最小的满足 nums[i] >= target 的 i
// 如果数组为空，或者所有数都 < target，则返回 nums.length
// 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]

// 闭区间写法
// 区间的定义: 表示我们需要知道这个区间内的元素和target的关系 ！！！！！！
private int lowerBound(int[] nums, int target) {
    int left = 0, right = nums.length - 1; // 闭区间 [left, right]
    while (left <= right) { // 区间不为空
        int mid = left + (right - left) / 2; //防止加法溢出 ！！！！！！
        if (nums[mid] < target) {
            left = mid + 1; // 范围缩小到 [mid+1, right]
        } else {
            right = mid - 1; // 范围缩小到 [left, mid-1]
        }
    }
    return left;
}
```

### 注意事项
1. 上述模板代码 lowerBound 返回的是最小的满足 nums[i] >= target 的 i（属于 >= 的情况）
2. 对于 >, <=, < 的其他三种情况，可以转换一下。比如 > target 可以转换为 >= target，以此类推。


## [2.搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/description)
思路：整个矩阵展平其实就是一个有序数组，直接用二分查找即可。     
代码：
```
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0;
        int right = n * m - 1;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(matrix[mid/n][mid%n] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        if(left == n * m || matrix[left/n][left%n] != target)
            return false;
        else return true;
    }
}
```


## [3.在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description)
思路：套用二分查找模板，详见代码注释。     
代码：
```
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int first = lowerbound(nums, target);
        //如果数组为空，或者所有数都 < target，则返回 nums.length
        //接着判断满足nums[i] >= target 的 i 所对应元素是否等于target
        if(first == nums.length || nums[first] != target)
            return new int[]{-1, -1};
        //能找到元素第一个位置，则继续查找最后一个位置
        int last = lowerbound(nums, target + 1);//target + 1
        return new int[]{first, last - 1};
    }

    public int lowerbound(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }
}
```

## [4.寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/)
思路：红色背景表示 false，即最小值左侧，蓝色背景表示 true，即最小值及其右侧。根据这一定义，n-1必然是蓝色。    
代码：
```
class Solution {
    public int findMin(int[] nums) {
        int n = nums.length;
        int left = 0;
        int right = n - 2;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < nums[n-1])
                right = mid - 1;
            else
                left = mid + 1;
        }
        return nums[left];
    }
}
```

## [5.搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/description)
思路：首先寻找旋转排序数组的最小值下标，然后分类讨论target处于哪一段。     
代码：
```
class Solution {
    public int search(int[] nums, int target) {
        //第一步：寻找旋转排序数组的最小值下标
        int n = nums.length;
        int left = 0;
        int right = n - 2;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < nums[n-1])
                right = mid -1;
            else
                left = mid + 1;
        }
        int minIdx = left; //最小值的下标
        //第二步，分类讨论target处于哪一段
        if(target <= nums[n-1]){ //可能位于右边这段
            int ans = lowerbound(nums, target, minIdx, n-1);
            if(ans == n || nums[ans] != target) return -1;
            else return ans;
        }else{ //可能位于左边这段
            int ans = lowerbound(nums, target, 0, minIdx-1);
            if(ans == minIdx || nums[ans] != target) return -1;
            else return ans;
        }
    }

    public int lowerbound(int[] nums, int target, int left, int right){
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }
}
```

# 栈

## [1.有效的括号](https://leetcode.cn/problems/valid-parentheses/description)
思路：维护一个栈，存放左括号。遍历到左括号则入栈，右括号则判断栈是否为空或者栈顶元素是否对应，返回false，否则配对，栈顶元素出栈。详见代码     
代码：
```
class Solution {
    public boolean isValid(String s) {
        //判断字符串长度是否是偶数
        char[] cArr = s.toCharArray();
        if(cArr.length % 2 == 1)
            return false;
        //维护一个栈，存放左括号
        List<Character> stack = new LinkedList<>();
        //哈希表，存放左右括号的对应关系
        Map<Character, Character> mp = new HashMap<>();
        mp.put(')', '(');
        mp.put(']', '[');
        mp.put('}', '{');
        //开始遍历
        for(int i = 0; i < cArr.length; i++){
            if(cArr[i] == '(' || cArr[i] == '[' || cArr[i] == '{'){//左括号直接入栈
                stack.addLast(cArr[i]);
            }else{ //右括号
                //判断栈是否为空或者栈顶元素是否对应
                if(stack.isEmpty()) return false;
                if(stack.getLast() != mp.get(cArr[i])) return false;
                stack.removeLast(); //配对，栈顶元素出栈
            }
        }
        return stack.isEmpty(); //最后判断栈是否为空
    }
}
```

## [2.最小栈](https://leetcode.cn/problems/min-stack/description)
思路：维护两个栈，层层对应。一个是正常的栈；一个是辅助最小栈，辅助最小栈的栈顶元素表示当前正常栈的最小值。    
代码：
```
class MinStack {

    public List<Integer> min_stack;
    public List<Integer> stack;
    public MinStack() {
        this.min_stack = new LinkedList<>(); //维护一个辅助最小栈
        this.min_stack.addLast(Integer.MAX_VALUE); //初始化辅助最小栈
        this.stack = new LinkedList<>(); //维护一个栈
    }
    
    public void push(int val) {
        stack.addLast(val); //入栈
        if(min_stack.getLast() > val) //层层对应，维护辅助最小栈
            min_stack.addLast(val);
        else
            min_stack.addLast(min_stack.getLast());
    }
    
    public void pop() {
        stack.removeLast();
        min_stack.removeLast();
    }
    
    public int top() {
        return stack.getLast();
    }
    
    public int getMin() {
        return min_stack.getLast();
    }
}
```

## [3.每日温度](https://leetcode.cn/problems/daily-temperatures/description/)
思路：单调栈题目。从右到左，维护一个单调栈，栈顶元素就是当前元素的下一个更大值。  
代码：
```
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        List<Integer> stack = new LinkedList<>(); //维护一个单调栈
        int[] ans = new int[temperatures.length];
        for(int i = temperatures.length - 1; i >= 0; i--){
            //维护单调栈
            while(!stack.isEmpty() && temperatures[stack.getLast()] <= temperatures[i]){
                stack.removeLast();
            }
            if(stack.isEmpty()){ //若栈为空，说明右边没有比他大的值
                ans[i] = 0;
                stack.addLast(i); //当前元素下标入栈
            }
            else{ //若栈顶不为空，则栈顶元素就是下一个更高温度
                ans[i] = stack.getLast() - i;
                stack.addLast(i); //当前元素下标入栈
            }
        }
        return ans;
    }
}
```

## [4.字符串解码](https://leetcode.cn/problems/decode-string/description)
思路：思路见Krahets。StringBuilder用append拼接字符串，toString转化为String  
代码：
```
class Solution {
    public String decodeString(String s) {
        char[] cArr = s.toCharArray();
        List<String> stack_res = new LinkedList<>();
        List<Integer> stack_mul = new LinkedList<>();
        StringBuilder res = new StringBuilder();; //记录左括号前的子串
        Integer mul = 0; //记录左括号前的数字
        for(int i = 0; i < cArr.length; i++){
            char c = cArr[i];
            if('0' <= c && c <= '9') //为数字
                mul = mul * 10 + (c - '0');
            else if('a' <= c && c <= 'z') //字母
                res.append(c);
            else if(c == '['){ //左括号
                stack_mul.addLast(mul); //入栈
                stack_res.addLast(res.toString()); //入栈
                res = new StringBuilder(); //置0置空
                mul = 0; //置0置空
            }
            else{ //右括号
                StringBuilder temp = new StringBuilder();
                Integer new_mul = stack_mul.removeLast();//出栈
                for(int j = 0; j < new_mul; j++){ //拼接字符串
                    temp.append(res);
                }
                res = new StringBuilder(stack_res.removeLast() + temp);
            }
        }
        return res.toString();
    }
}
```

# 贪心算法

## [1.买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description)
思路：for循环，记录截止当前买入的最低股票值，并计算在第i天卖出可以获取的最大利润。
代码：
```
class Solution {
    public int maxProfit(int[] prices) {
        int min_value = 10001; //记录截止当前买入的最低股票值
        int ans = 0; //可以获取的最大利润
        //for循环在第i天卖出，查找可以获取的最大利润
        for(int i = 0; i < prices.length; i++){
            if(prices[i] > min_value){ //可以卖出
                ans = Math.max(ans, prices[i] - min_value); //更新答案
            }
            min_value = Math.min(min_value, prices[i]); //更新最小值
        }
        return ans;
    }
}
```

## [2.跳跃游戏](https://leetcode.cn/problems/jump-game/description)
思路：依次遍历数组中的每一个位置，并实时维护 最远可以到达的位置；如果当前位置已超出目前可到达的最远距离，则返回false。
代码：
```
class Solution {
    public boolean canJump(int[] nums) {
        int maxLength = 0; //目前可到达的最远距离
        for(int i = 0; i < nums.length; i++){
            if(i > maxLength) return false; //当前位置已超出目前可到达的最远距离
            maxLength = Math.max(maxLength, i + nums[i]); //更新目前可到达的最远距离
        }
        return true;
    }
}
```

## [3.跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/description)
思路：参考灵神的思路。想象你在一段一段地建桥，每次在你已建的桥上寻找下一次可建的最远桥，当走到已建桥的尽头就进行下一次的建桥。最后建了几段桥，就是最小的跳跃步数。
代码：
```
class Solution {
    public int jump(int[] nums) {
        int n = nums.length;
        int curBridge = 0; //当前桥的右端点
        int nextBridge = 0; //下一次建桥的最远位置
        int step = 0; //最小跳跃次数
        for(int i = 0; i < n - 1; i++){ //注意不用遍历最后一个位置
            nextBridge = Math.max(nextBridge, i + nums[i]);
            if(i == curBridge){ //走到桥的尽头
                curBridge = nextBridge; //建桥
                step++;//步数加1
            }
        }
        return step;
    }
}
```

## [4.划分字母区间](https://leetcode.cn/problems/partition-labels/description)
思路：参考灵神的思路。首先遍历字符串s计算每个字符的最右下标。之后维护一个区间[start, end]，重新遍历s，每次更新end = max(end, 当前字符最右下标)，然后判断当前i是否已经走到end，若是则划分该区间，重复此过程。   
代码：
```
class Solution {
    public List<Integer> partitionLabels(String s) {
        Map<Character, Integer> mp = new HashMap<>();//记录每个字符的最右下标
        char[] arr = s.toCharArray();
        for(int i = 0; i < arr.length; i++){
            mp.put(arr[i], i);
        }
        //开始划分字母区间
        List<Integer> ans = new ArrayList<>();
        int start = 0, end = 0; //记录当前待划分区间
        for(int i = 0; i < arr.length; i++){
            end = Math.max(end, mp.get(arr[i])); //更新区间右边界
            if(i == end){ //表示区间内的字符后面不会再出现了，可以进行划分
                ans.add(end - start + 1);
                start = end + 1;
                end = start;
            }
        }
        return ans;
    }
}
```

# 动态规划

## [1.爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/)
思路：台阶i方法数=台阶i-1方法数（再爬一个台阶）+台阶i-2方法数（再爬二个台阶）     
代码：
```
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
}
```

## [2.杨辉三角](https://leetcode.cn/problems/pascals-triangle/description)
思路：每一行左对齐。各行的首尾都是1，中间部分则通过dp继续计算。  
代码：
```
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> line = new ArrayList<>();
        line.add(1);
        ans.add(line);
        for(int i = 1; i < numRows; i++){
            line = ans.get(i-1);
            List<Integer> new_line = new ArrayList<>();
            new_line.add(1);
            for(int j = 1; j < line.size(); j++){
                new_line.add(line.get(j-1) + line.get(j));
            }
            new_line.add(1);
            ans.add(new_line);
        }
        return ans;
    }
}
```

## [3.打家劫舍](https://leetcode.cn/problems/house-robber/description)
思路：选或者不选。前i号房屋的最高金额 = max(选：前i-2号房屋的最高金额+i号房屋的金额，不选：前i-1号房屋的最高金额)，得到dp递推公式。    
代码：  
```
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = nums[0];
        for(int i = 2; i <= n; i++){
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i-1]);
        }
        return dp[n];
    }
}
```

## [4.完全平方数](https://leetcode.cn/problems/perfect-squares/description/)
思路：状态转移方程 dp[i][j] = min(dp[i-1][j], dp[i][j-i*i]+1)，套用完全背包一维模板。   
代码：
```
class Solution {
    public int numSquares(int n) {
        int x = (int)Math.floor(Math.sqrt(n)); //得到小于等于n的最大完全平方数的开方
        int[] dp = new int[n+1]; //dp[i][j]表示考虑前i个数，和为j的完全平方数的最少数量
        for(int i = 1; i <= n; i++) dp[i] = Integer.MAX_VALUE; //初始化
        for(int i = 1; i <= x; i++){
            for(int j = i*i; j <= n; j++){
                dp[j] = Math.min(dp[j], dp[j-i*i] + 1);
            }
        }
        return dp[n];
    }
}
```

## [5.零钱兑换](https://leetcode.cn/problems/coin-change/description/)
思路：状态转移方程：f[i][j] = min(f[i-1][j], f[i][j-val]+1)，套用完全背包一维模板。   
代码：
```
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        for(int i = 1; i <= amount; i++) dp[i] = 10001; //初始化
        for(int i = 1; i <= n; i++){
            for(int j = coins[i-1]; j <= amount; j++){
                dp[j] = Math.min(dp[j], dp[j-coins[i-1]] + 1);
            }
        }
        return dp[amount] == 10001 ? -1 : dp[amount];
    }
}
```

## [6.最长递增子序列 LIS](https://leetcode.cn/problems/longest-increasing-subsequence/description/)
思路一：dp[i]表示以nums[i]元素结尾的最长递增子序列长度。状态转移方程为dp[i] = max{dp[j]}+1，其中j<i且nums[j] < nums[i]。      
代码：
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n]; //dp[i]表示以nums[i]元素结尾的最长递增子序列长度。
        //状态转移方程为dp[i] = max{dp[j]}+1，其中j<i且nums[j] < nums[i]
        int ans = 0;
        for(int i = 0; i < n; i++)
        {
            int max = 0;
            for(int j = 0; j < i; j++)
            {
                if(nums[j] < nums[i] && dp[j] > max)
                    max = dp[j];
            }
            dp[i] = max + 1;
            if(dp[i] > ans) ans = dp[i];
        }
        return ans;
    }
}
```

思路二：贪心 + 二分查找。考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。基于上面的贪心思路，我们维护一个数组 d[i] ，表示长度为 i 的最长上升子序列的末尾元素的最小值，用 len 记录目前最长上升子序列的长度，起始时 len 为 1，d[1]=nums[0]。依次遍历数组 nums 中的每个元素，并更新数组 d 和 len 的值。如果 nums[i]>d[len] 则更新 len=len+1，并将nums[i]添加到d数组末尾。否则在 d[1…len]（二分查找闭区间写法，下标从1到len）中找到第一个大于等于nums[i]的下标k，将d[k]覆盖为nums[i]。最终返回len即为结果。

代码：
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] d = new int[nums.length + 1]; //d[i]表示长度为i的严格递增子序列的末尾最小元素
        int len = 1; //记录目前最长上升子序列的长度
        d[len] = nums[0];
        for(int i = 1; i < nums.length; i++){
            if(nums[i] > d[len]){
                len++;
                d[len] = nums[i];
            }else{
                int idx = lowerBound(d, 1, len, nums[i]);
                d[idx] = nums[i];
            }
        }
        return len;
    }

    public int lowerBound(int[] nums, int left, int right, int target){
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }
}
```

## [7.分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/)
思路：问题转换为从数组nums选择一些数，使得其和恰好为sum/2。0-1背包问题，状态转移方程为:dp[i][j] = dp[i-1][j] || dp[i-1][j-val]。     
代码：
```
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for(int x : nums) sum += x; //计算总和
        if(sum % 2 == 1) return false; //如果总和为奇数，返回false
        //状态转移方程为：dp[i][j] = dp[i-1][j] | dp[i-1][j-v]
         boolean[] dp = new boolean[sum/2+1]; //dp[i][j]表示前i个数中有可以组成和恰好为j的子集。值为0或者1
        dp[0] = true; //初始化
        for(int i = 1; i <= n; i++){
            for(int j = sum/2; j >= nums[i-1]; j--){
                dp[j] = dp[j] | dp[j-nums[i-1]];
            }
            if(dp[sum/2]) return true; //加多这一行，提前判断是否找到满足sum/2的子集
        }
        return dp[sum/2];
    }
}
```

## [8.单词拆分](https://leetcode.cn/problems/word-break/description)
思路：首先用哈希表记录字典中的单词。然后for循环i，dp[i]表示s前i个字符是否由多个单词拼接而成。状态转移方程dp[i] = dp[j] && contains(s[j..i-1])，其中contains(s[j..i−1]) 表示子串 s[j..i−1] 是否出现在字典中。    
代码：
```
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> mp = new HashSet<>(); //哈希表用于记录单词是否出现
        for(String str : wordDict){
            mp.add(str);
        }
        int n = s.length();
        boolean[] dp = new boolean[n+1]; //dp[i]表示s前i个字符是否由多个单词拼接而成
        dp[0] = true; //初始化
        for(int i = 1; i <= n; i++){
            //状态转移方程dp[i] = dp[j] && contains(s[j..i-1])
            //其中 contains(s[j..i−1]) 表示子串 s[j..i−1] 是否出现在字典中
            for(int j = i-1; j >= 0; j--){ //子串的起始下标
                if(dp[j] && mp.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

## [9.乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description)
思路：本题是[53. 最大子数组和]的乘法版本。首先用两个数组fmax和fmin记录以下标i结尾的子数组最大/最小乘积（分别代表正负，如果下一元素是负数，负负得正可以得到最大值），for循环遍历，状态转移方程为min(fmin[i-1] * nums[i], fmax[i-1] * nums[i], nums[i])和fmax[i] = max(fmin[i-1] * nums[i], fmax[i-1] * nums[i], nums[i])。最后返回MAX（fmaxx[i]）。    
代码：
```
class Solution {
    public int maxProduct(int[] nums) {
        int[] fmin = new int[nums.length]; //fmin[i]表示以下标i结尾的子数组最小乘积
        int[] fmax = new int[nums.length]; //fmax[i]表示以下标i结尾的子数组最大乘积
        fmin[0] = nums[0];
        fmax[0] = nums[0];
        for(int i = 1; i < nums.length; i++){
            fmin[i] = Math.min(Math.min(fmin[i-1] * nums[i], fmax[i-1] * nums[i]), nums[i]);
            fmax[i] = Math.max(Math.max(fmin[i-1] * nums[i], fmax[i-1] * nums[i]), nums[i]);
        }
        int ans = Integer.MIN_VALUE;
        for(int x : fmax){
            ans = Math.max(ans, x);
        }
        return ans;
    }
}
```



# 模拟题

## [1.字符串相加](https://leetcode.cn/problems/add-strings/description/)
思路：和链表的两数相加类似。两个字符串从低位开始相加，直到最长数走完和进位为空为止。     
代码：
```
class Solution {
    public String addStrings(String num1, String num2) {
        int l1 = num1.length() - 1; //从字符串右边（低位）开始计算
        int l2 = num2.length() - 1; //从字符串右边（低位）开始计算
        int carry = 0; //进位
        StringBuilder ans = new StringBuilder();
        while(l1 >= 0 || l2 >= 0 || carry != 0){
            int a = l1 >= 0 ? num1.charAt(l1) - '0' : 0;
            int b = l2 >= 0 ? num2.charAt(l2) - '0' : 0;
            int val = a + b + carry;
            ans.append(val % 10);
            carry = val / 10;
            l1--;
            l2--;
        }
        return ans.reverse().toString();
    }
}
```

## [2.字符串相乘](https://leetcode.cn/problems/multiply-strings/description/)
思路：用num2的每一位去乘以num1，然后补齐后缀0，再相加。     
代码：
```
class Solution {
    public String multiply(String num1, String num2) {
        if(num1.equals("0") || num2.equals("0")) return "0"; //如果有一个为0则直接返回0
        String ans = "0";
        for(int i = num2.length()-1; i >= 0; i--){ //用num2的每一位去乘以num1，再相加。
            StringBuilder temp = new StringBuilder(mul(num1, num2.charAt(i)));
            //补充temp的后缀0
            for(int j = 0; j < num2.length()-1-i; j++)
                temp.append("0");
            ans = addStrings(ans, temp.toString());
        }
        return ans;
    }

    //1位数乘以一个数
    public String mul(String num, char c){
        int l = num.length() - 1;
        int carry = 0;
        StringBuilder ans = new StringBuilder();
        while(l >= 0 || carry != 0){
            int a = l >= 0? num.charAt(l) - '0' : 0;
            int val = a * (c - '0') + carry;
            ans.append(val % 10);
            carry = val / 10;
            l--;
        }
        return ans.reverse().toString();
    }

    //两数相加
    public String addStrings(String num1, String num2) {
        int l1 = num1.length() - 1; //从字符串右边（低位）开始计算
        int l2 = num2.length() - 1; //从字符串右边（低位）开始计算
        int carry = 0; //进位
        StringBuilder ans = new StringBuilder();
        while(l1 >= 0 || l2 >= 0 || carry != 0){
            int a = l1 >= 0 ? num1.charAt(l1) - '0' : 0;
            int b = l2 >= 0 ? num2.charAt(l2) - '0' : 0;
            int val = a + b + carry;
            ans.append(val % 10);
            carry = val / 10;
            l1--;
            l2--;
        }
        return ans.reverse().toString();
    }
}
```


# 多维动态规划

## [1.不同路径](https://leetcode.cn/problems/unique-paths/description)
思路：状态转移方程为dp[i][j] = dp[i-1][j] + dp[i][j-1];     
代码：
```
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m+1][n+1];
        dp[0][1] = 1; //初始化，用于计算dp[1][1]
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m][n];
    }
}
```

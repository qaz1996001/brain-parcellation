CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `xxx_dm_part_bt`
--

DROP TABLE IF EXISTS `xxx_dm_part_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `xxx_dm_part_bt` (
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(16) DEFAULT NULL,
  `route_id` varchar(24) NOT NULL DEFAULT 'ROUTE-ZZZ-01' COMMENT 'for user not need to create each part / step record -- 2020/03/26 lkchena',
  `rd_flag` int(11) NOT NULL DEFAULT 0 COMMENT '2020/04/14 lkchena',
  `mold_id` varchar(16) DEFAULT NULL COMMENT '??? 1 part vs 1 mold ???',
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `lot_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'total unit count 2018/06/06 lkchena\nin one cart / plate on the stb tool 2020/04/22 lkchena',
  `powder_type` varchar(16) DEFAULT NULL COMMENT 'default powder, maybe change when shaping 2018/05/18 lkchena',
  `weight_above` decimal(3,1) DEFAULT NULL,
  `weight_below` decimal(3,1) DEFAULT NULL,
  `density_above` decimal(3,1) DEFAULT NULL,
  `density_below` decimal(3,1) DEFAULT NULL,
  `hardness` decimal(3,1) DEFAULT NULL,
  `material` varchar(24) DEFAULT NULL COMMENT '原料',
  `layout_doc` varchar(254) DEFAULT NULL COMMENT 'part layout design document file path',
  `note` varchar(128) DEFAULT NULL,
  `cnt_cart_1` int(11) NOT NULL DEFAULT 0 COMMENT '2020/03/06 lkchena\ncart / box default size  <-- different by PET plastic\n\ncnt_cart_1,cnt_box_1: forming default size\ncnt_cart_2,cnt_box_2: sintering default size\ncnt_cart_3,cnt_box_3: \ncnt_cart_4,cnt_box_4: ',
  `cnt_cart_2` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_cart_3` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_cart_4` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_box_1` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_box_2` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_box_3` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `cnt_box_4` int(11) NOT NULL DEFAULT 0 COMMENT 'see cnt_cart_1 definiton -- 2020/03/06 lkchena',
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `xxx_dm_part_bt`
--

LOCK TABLES `xxx_dm_part_bt` WRITE;
/*!40000 ALTER TABLE `xxx_dm_part_bt` DISABLE KEYS */;
INSERT INTO `xxx_dm_part_bt` VALUES ('IAC69-40040 SILDER','IAC69-40040 SILD','ZZZ-01',0,'SILDER-0235','SANDEN',NULL,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.186288'),('SC33-GEAR-01','SC33-GEAR-01','ZZZ-01',0,'FF-1234','BKM1',NULL,0,'S06,41',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.187290'),('SC34-GEAR-02','SC34-GEAR-02','ZZZ-01',0,'FF-1200','Custom Real',NULL,0,'S06,41',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.187290'),('SC37-GEAR-03','SC37-GEAR-03','ZZZ-01',0,'FF-1238','Custom',NULL,0,'S06,40',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.188292'),('WCE42-20025','WCE42-20025','ZZZ-01',0,'MOLD-STD-002','HELLWE',NULL,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.188292'),('ZZZ-01','ZZZ-01','ZZZ-01',0,'ZZZ-01','ZZZ-01',NULL,0,'powder1',23.0,24.0,8.2,7.9,10.2,NULL,NULL,NULL,896,1400,0,0,28,25,0,0,'SYS','2020/03/07 16:53:47',NULL,'2020-04-22 00:03:57.189293');
/*!40000 ALTER TABLE `xxx_dm_part_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:29
